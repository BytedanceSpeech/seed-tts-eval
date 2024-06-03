import sys
import h5py
import soundfile as sf
import fire
import math
import yamlargparse
import numpy as np
from torch.utils.data import DataLoader
import torch
from utils.utils import parse_config_or_kwargs
from utils.dataset import DiarizationDataset
from models.models import TransformerDiarization
from scipy.signal import medfilt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from utils.kaldi_data import KaldiData


def get_cl_sil(args, acti, cls_num):
    n_chunks = len(acti)
    mean_acti = np.array([np.mean(acti[i], axis=0)
                         for i in range(n_chunks)]).flatten()
    n = args.num_speakers
    sil_spk_th = args.sil_spk_th

    cl_lst = []
    sil_lst = []
    for chunk_idx in range(n_chunks):
        if cls_num is not None:
            if args.num_speakers > cls_num:
                mean_acti_bi = np.array([mean_acti[n * chunk_idx + s_loc_idx]
                                        for s_loc_idx in range(n)])
                min_idx = np.argmin(mean_acti_bi)
                mean_acti[n * chunk_idx + min_idx] = 0.0

        for s_loc_idx in range(n):
            a = n * chunk_idx + (s_loc_idx + 0) % n
            b = n * chunk_idx + (s_loc_idx + 1) % n
            if mean_acti[a] > sil_spk_th and mean_acti[b] > sil_spk_th:
                cl_lst.append((a, b))
            else:
                if mean_acti[a] <= sil_spk_th:
                    sil_lst.append(a)

    return cl_lst, sil_lst


def clustering(args, svec, cls_num, ahc_dis_th, cl_lst, sil_lst):
    org_svec_len = len(svec)
    svec = np.delete(svec, sil_lst, 0)

    # update cl_lst idx
    _tbl = [i - sum(sil < i for sil in sil_lst) for i in range(org_svec_len)]
    cl_lst = [(_tbl[_cl[0]], _tbl[_cl[1]]) for _cl in cl_lst]

    distMat = distance.cdist(svec, svec, metric='euclidean')
    for cl in cl_lst:
        distMat[cl[0], cl[1]] = args.clink_dis
        distMat[cl[1], cl[0]] = args.clink_dis

    clusterer = AgglomerativeClustering(
            n_clusters=cls_num,
            affinity='precomputed',
            linkage='average',
            distance_threshold=ahc_dis_th)
    clusterer.fit(distMat)

    if cls_num is not None:
        print("oracle n_clusters is known")
    else:
        print("oracle n_clusters is unknown")
        print("estimated n_clusters by constraind AHC: {}"
              .format(len(np.unique(clusterer.labels_))))
        cls_num = len(np.unique(clusterer.labels_))

    sil_lab = cls_num
    insert_sil_lab = [sil_lab for i in range(len(sil_lst))]
    insert_sil_lab_idx = [sil_lst[i] - i for i in range(len(sil_lst))]
    print("insert_sil_lab : {}".format(insert_sil_lab))
    print("insert_sil_lab_idx : {}".format(insert_sil_lab_idx))
    clslab = np.insert(clusterer.labels_,
                       insert_sil_lab_idx,
                       insert_sil_lab).reshape(-1, args.num_speakers)
    print("clslab : {}".format(clslab))

    return clslab, cls_num


def merge_act_max(act, i, j):
    for k in range(len(act)):
        act[k, i] = max(act[k, i], act[k, j])
        act[k, j] = 0.0
    return act


def merge_acti_clslab(args, acti, clslab, cls_num):
    sil_lab = cls_num
    for i in range(len(clslab)):
        _lab = clslab[i].reshape(-1, 1)
        distM = distance.cdist(_lab, _lab, metric='euclidean').astype(np.int64)
        for j in range(len(distM)):
            distM[j][:j] = -1
        idx_lst = np.where(np.count_nonzero(distM == 0, axis=1) > 1)
        merge_done = []
        for j in idx_lst[0]:
            for k in (np.where(distM[j] == 0))[0]:
                if j != k and clslab[i, j] != sil_lab and k not in merge_done:
                    print("merge : (i, j, k) == ({}, {}, {})".format(i, j, k))
                    acti[i] = merge_act_max(acti[i], j, k)
                    clslab[i, k] = sil_lab
                    merge_done.append(j)

    return acti, clslab


def stitching(args, acti, clslab, cls_num):
    n_chunks = len(acti)
    s_loc = args.num_speakers
    sil_lab = cls_num
    s_tot = max(cls_num, s_loc-1)

    # Extend the max value of s_loc_idx to s_tot+1
    add_acti = []
    for chunk_idx in range(n_chunks):
        zeros = np.zeros((len(acti[chunk_idx]), s_tot+1))
        if s_tot+1 > s_loc:
            zeros[:, :-(s_tot+1-s_loc)] = acti[chunk_idx]
        else:
            zeros = acti[chunk_idx]
        add_acti.append(zeros)
    acti = np.array(add_acti)

    out_chunks = []
    for chunk_idx in range(n_chunks):
        # Make sloci2lab_dct.
        # key: s_loc_idx
        # value: estimated label by clustering or sil_lab
        cls_set = set()
        for s_loc_idx in range(s_tot+1):
            cls_set.add(s_loc_idx)

        sloci2lab_dct = {}
        for s_loc_idx in range(s_tot+1):
            if s_loc_idx < s_loc:
                sloci2lab_dct[s_loc_idx] = clslab[chunk_idx][s_loc_idx]
                if clslab[chunk_idx][s_loc_idx] in cls_set:
                    cls_set.remove(clslab[chunk_idx][s_loc_idx])
                else:
                    if clslab[chunk_idx][s_loc_idx] != sil_lab:
                        raise ValueError
            else:
                sloci2lab_dct[s_loc_idx] = list(cls_set)[s_loc_idx-s_loc]

        # Sort by label value
        sloci2lab_lst = sorted(sloci2lab_dct.items(), key=lambda x: x[1])

        # Select sil_lab_idx
        sil_lab_idx = None
        for idx_lab in sloci2lab_lst:
            if idx_lab[1] == sil_lab:
                sil_lab_idx = idx_lab[0]
                break
        if sil_lab_idx is None:
            raise ValueError

        # Get swap_idx
        # [idx of label(0), idx of label(1), ..., idx of label(s_tot)]
        swap_idx = [sil_lab_idx for j in range(s_tot+1)]
        for lab in range(s_tot+1):
            for idx_lab in sloci2lab_lst:
                if lab == idx_lab[1]:
                    swap_idx[lab] = idx_lab[0]

        print("swap_idx {}".format(swap_idx))
        swap_acti = acti[chunk_idx][:, swap_idx]
        swap_acti = np.delete(swap_acti, sil_lab, 1)
        out_chunks.append(swap_acti)

    return out_chunks


def prediction(num_speakers, net, wav_list, chunk_len_list):
    acti_lst = []
    svec_lst = []
    len_list = []

    with torch.no_grad():
        for wav, chunk_len in zip(wav_list, chunk_len_list):
            wav = wav.to('cuda')
            outputs = net.batch_estimate(torch.unsqueeze(wav, 0))
            ys = outputs[0]

            for i in range(num_speakers):
                spkivecs = outputs[i+1]
                svec_lst.append(spkivecs[0].cpu().detach().numpy())

            acti = ys[0][-chunk_len:].cpu().detach().numpy()
            acti_lst.append(acti)
            len_list.append(chunk_len)

    acti_arr = np.concatenate(acti_lst, axis=0)  # totol_len x num_speakers
    svec_arr = np.stack(svec_lst)                # (chunk_num x num_speakers) x emb_dim
    len_arr = np.array(len_list)                 # chunk_num

    return acti_arr, svec_arr, len_arr

def cluster(args, conf, acti_arr, svec_arr, len_arr):

    acti_list = []
    n_chunks = len_arr.shape[0]
    start = 0
    for i in range(n_chunks):
        chunk_len = len_arr[i]
        acti_list.append(acti_arr[start: start+chunk_len])
        start += chunk_len
    acti = np.array(acti_list)
    svec = svec_arr

    # initialize clustering setting
    cls_num = None
    ahc_dis_th = args.ahc_dis_th
    # Get cannot-link index list and silence index list
    cl_lst, sil_lst = get_cl_sil(args, acti, cls_num)

    n_samples = n_chunks * args.num_speakers - len(sil_lst)
    min_n_samples = 2
    if cls_num is not None:
        min_n_samples = cls_num

    if n_samples >= min_n_samples:
        # clustering (if cls_num is None, update cls_num)
        clslab, cls_num =\
             clustering(args, svec, cls_num, ahc_dis_th, cl_lst, sil_lst)
        # merge
        acti, clslab = merge_acti_clslab(args, acti, clslab, cls_num)
        # stitching
        out_chunks = stitching(args, acti, clslab, cls_num)
    else:
        out_chunks = acti

    outdata = np.vstack(out_chunks)
    # Saving the resuts
    return outdata

def make_rttm(args, conf, cluster_data):
    args.frame_shift = conf['model']['frame_shift']
    args.subsampling = conf['model']['subsampling']
    args.sampling_rate = conf['dataset']['sampling_rate']

    with open(args.out_rttm_file, 'w') as wf:
        a = np.where(cluster_data > args.threshold, 1, 0)
        if args.median > 1:
            a = medfilt(a, (args.median, 1))
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(fmt.format(
                    args.session,
                    s * args.frame_shift * args.subsampling / args.sampling_rate,
                    (e - s) * args.frame_shift * args.subsampling / args.sampling_rate,
                    args.session + "_" + str(spkid)), file=wf)

def main(args):
    conf = parse_config_or_kwargs(args.config_path)
    num_speakers = conf['dataset']['num_speakers']
    args.num_speakers = num_speakers

    # Prepare model
    model_parameter_dict = torch.load(args.model_init)['model']
    model_all_n_speakers = model_parameter_dict["embed.weight"].shape[0]
    conf['model']['all_n_speakers'] = model_all_n_speakers
    net = TransformerDiarization(**conf['model'])
    net.load_state_dict(model_parameter_dict, strict=False)
    net.eval()
    net = net.to("cuda")

    audio, sr = sf.read(args.wav_path, dtype="float32")
    audio_len = audio.shape[0]
    chunk_size, frame_shift, subsampling = conf['dataset']['chunk_size'], conf['model']['frame_shift'], conf['model']['subsampling']
    scale_ratio = int(frame_shift * subsampling)
    chunk_audio_size = chunk_size * scale_ratio
    wav_list, chunk_len_list = [], []
    for i in range(0, math.ceil(1.0 * audio_len / chunk_audio_size)):
        start, end = i*chunk_audio_size, (i+1)*chunk_audio_size
        if end > audio_len:
            chunk_len_list.append(int((audio_len-start) / scale_ratio))
            end = audio_len
            start = max(0, audio_len - chunk_audio_size)
        else:
            chunk_len_list.append(chunk_size)
        wav_list.append(audio[start:end])
    wav_list = [torch.from_numpy(wav).float() for wav in wav_list]

    acti_arr, svec_arr, len_arr = prediction(num_speakers, net, wav_list, chunk_len_list)
    cluster_data = cluster(args, conf, acti_arr, svec_arr, len_arr)
    make_rttm(args, conf, cluster_data)

if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser(description='decoding')
    parser.add_argument('--wav_path',
                        help='the input wav path',
                        default="tmp/mix_0000496.wav")
    parser.add_argument('--config_path',
                        help='config file path',
                        default="config/infer_est_nspk1.yaml")
    parser.add_argument('--model_init',
                        help='model initialize path',
                        default="")
    parser.add_argument('--sil_spk_th', default=0.05, type=float)
    parser.add_argument('--ahc_dis_th', default=1.0, type=float)
    parser.add_argument('--clink_dis', default=1.0e+4, type=float)
    parser.add_argument('--session', default='Anonymous', help='the name of the output speaker')
    parser.add_argument('--out_rttm_file', default='out.rttm', help='the output rttm file')
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--median', default=25, type=int)


    args = parser.parse_args()
    main(args)
