import numpy as np
from scipy.ndimage.filters import gaussian_filter
from config import get_default_configuration


def get_coordinates(config, heatmaps, threshold=0.1):
    """
    第一步：根据heatmap寻找出构成肢体的关键点。
    Finds the coordinates x,y in the heatmaps.

    :param config: pose estimation configuration
    :param heatmaps: heatmaps
    :param threshold: threshold for the intensity value in the heatmap at the position of a peak
    :return: dictionary:
        { body_part_name:
            [(x ,y, score, id),
             (x ,y, score, id),
             ...
            ],
          ...
        }
    """
    all_peaks = dict()
    peak_counter = 0

    for part_meta in config.body_parts.values():
        hmap_orig = heatmaps[:, :, part_meta.heatmap_idx]
        hmap = gaussian_filter(hmap_orig, sigma=3)

        # 将热力图上下左右平移，找出极大值点
        hmap_right = np.zeros(hmap.shape)
        hmap_right[:, 1:] = hmap[:, :-1]
        hmap_left = np.zeros(hmap.shape)
        hmap_left[:, :-1] = hmap[:, 1:]
        hmap_down = np.zeros(hmap.shape)
        hmap_down[1:, :] = hmap[:-1, :]
        hmap_up = np.zeros(hmap.shape)
        hmap_up[:-1, :] = hmap[1:, :]

        peaks_binary = np.logical_and.reduce(
            (hmap >= hmap_right,
             hmap >= hmap_left,
             hmap >= hmap_down,
             hmap >= hmap_up,
             hmap > threshold))

        peaks = list(zip(np.nonzero(peaks_binary)[1],
                         np.nonzero(peaks_binary)[0]))

        sequence_numbers = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peak + (
            hmap_orig[peak[1], peak[0]],
            seq_num) for peak, seq_num in zip(peaks, sequence_numbers)]

        all_peaks[part_meta.body_part.name] = peaks_with_score_and_id
        peak_counter += len(peaks)

    return all_peaks

