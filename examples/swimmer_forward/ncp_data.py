""" swimmer: open-loop control of forward locomotion
preprocess forward simulation dataset for NCP network training
"""

from virtual_nematode.data.utils import subset_and_split


if __name__ == '__main__':
    subset_and_split(data_size=1, seq_len=16, load_name='forward.pt', save_name='forward_ncp.pt')
