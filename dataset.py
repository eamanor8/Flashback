# OK, TRANSFERRED 

import random
from enum import Enum

import torch
from torch.utils.data import Dataset


class Split(Enum):
    ''' Defines whether to split for train or test.
    '''
    TRAIN = 0
    TEST = 1
    ALL = 2

class Usage(Enum):
    '''
    Each user has a different amount of sequences. The usage defines
    how many sequences are used:

    MAX: each sequence of any user is used (default)
    MIN: only as many as the minimal user has
    CUSTOM: up to a fixed amount if available.

    The unused sequences are discarded. This setting applies after the train/test split.
    '''

    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


class PoiDataset(Dataset):

    '''
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
    "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users: if an active user
    runs out of locations we replace him with a new one. When there are no unused users available
    we reuse already processed ones. This happens if a single user was way more active than the average user.
    The batch guarantees that each sequence of each user was processed at least once.

    This data management has the implication that some sequences might be processed twice (or more) per epoch.
    During trainig you should call PoiDataset::shuffle_users before the start of a new epoch. This
    leads to more stochastic as different sequences will be processed twice.
    During testing you *have to* keep track of the already processed users.

    Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user.
    We work with a 80/20 train/test spilt, where test check-ins are strictly after training checkins.
    To obtain at least one test sequence with label we require any user to have at least (5*<sequence-length>+1) checkins in total.
    '''

    def reset(self):
        # reset training state:
        self.next_user_idx = 0 # current user index to add
        self.active_users = [] # current active users
        self.active_user_seq = [] # current active users sequences
        self.user_permutation = [] # shuffle users during training

        # set active users:
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)

        # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)


    def shuffle_users(self):
        random.shuffle(self.user_permutation)
        # reset active users:
        self.next_user_idx = 0  #* this is a pointer into the self.user_permutation (reshuffled in each epoch)
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(self.user_permutation[i])
            self.active_user_seq.append(0)

    def __init__(self, users, times, coords, locs, sequence_length, batch_size, split, usage, loc_count, custom_seq_count):
        #* Inputs:
        #* times: list of (one list per user), from t=0 to t=T
        #* coords: list of (one list per user), from t=0 to t=T
        #* locs: list of (one list per user), from t=0 to t=T

        self.users = users    #* list of (one user ID per user)
        self.times = times    #* list of (one list per user)  # These are the times of Xs, (will be set below) from t=0 to t=T-1
        self.coords = coords  #* list of (one list per user)  # These are the coords of Xs, (will be set below) from t=0 to t=T-1
        self.locs = locs      #* list of (one list per user)  # These are the Xs, (will be set below) from t=0 to t=T-1
        self.labels = []      #* list of (one list per user)  # These are the Ys, from t=1 to t=T, set as label0 to labelT-1
        self.lbl_times = []   #* list of (one list per user)  # These are the times of Ys, from t=1 to t=T, set as label0 to labelT-1
        self.lbl_coords = []  #* list of (one list per user)  # These are the coords of Ys, from t=1 to t=T, set as label0 to labelT-1

        self.sequences = []             #* list of (one list of full--length-20--sequences per user). loc IDs. MAY SKIP SOME DATA POINTS
        self.sequences_times = []       #* list of (one list of full--length-20--sequences per user). MAY SKIP SOME DATA POINTS
        self.sequences_coords = []      #* list of (one list of full--length-20--sequences per user). MAY SKIP SOME DATA POINTS
        self.sequences_labels = []      #* list of (one list of full--length-20--sequences per user). loc IDs. MAY SKIP SOME DATA POINTS
        self.sequences_lbl_times = []   #* list of (one list of full--length-20--sequences per user). MAY SKIP SOME DATA POINTS
        self.sequences_lbl_coords = []  #* list of (one list of full--length-20--sequences per user). MAY SKIP SOME DATA POINTS
        self.sequences_count = []       #* list of (one number of full--length-20--sequences per user)
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.batch_size = batch_size
        self.loc_count = loc_count  #* number of locations in total across all users
        self.custom_seq_count = custom_seq_count  #* ???; not used

        self.reset()

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i # Tensor of location IDs (used for alignment).

        # Open a file to write the output
        with open("./100_users/UserCheckins.txt", "w") as f:    
            # align labels to locations (shift by one)
            for i, loc in enumerate(locs):
                self.locs[i] = loc[:-1] # Remove last location
                self.labels.append(loc[1:]) # Adds the shifted locations (starting from index 1) as labels
                # adapt time and coords:
                self.lbl_times.append(self.times[i][1:])
                self.lbl_coords.append(self.coords[i][1:])
                self.times[i] = self.times[i][:-1]
                self.coords[i] = self.coords[i][:-1]

                # Write the check-ins for each user to the file
                f.write(f"User {i} check-ins:\n")
                f.write(f"  Times: {self.times[i]}\n")
                f.write(f"  Coords: {self.coords[i]}\n")
                f.write(f"  Locs: {self.locs[i]}\n\n")

                # Write the sorted check-ins for each user to the file
                f.write(f"User {i} sorted check-ins:\n")
                sorted_checkins = sorted(zip(self.times[i], self.coords[i], self.locs[i]))
                f.write(f"  Sorted check-ins: {sorted_checkins}\n")
                
                # Extract and write locations from sorted check-ins as a list
                sorted_locations = [loc for _, _, loc in sorted_checkins]
                f.write(f"  Sorted Locations: {sorted_locations}\n\n\n")
                

        # split to training / test phase:  #* this iterates over users i
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(zip(self.times, self.coords, self.locs, self.labels, self.lbl_times, self.lbl_coords)):
            train_thr = int(len(loc) * 0.8)
            # print(f"i={i} |-> train_thr={train_thr}")  #* DEBUG SET A
            if (split == Split.TRAIN):
                self.times[i] = time[:train_thr]
                self.coords[i] = coord[:train_thr]
                self.locs[i] = loc[:train_thr]
                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]
            if (split == Split.TEST):
                self.times[i] = time[train_thr:]
                self.coords[i] = coord[train_thr:]
                self.locs[i] = loc[train_thr:]
                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]
            if (split == Split.ALL):
                self.times[i] = time
                self.coords[i] = coord
                self.locs[i] = loc
                self.labels[i] = label
                self.lbl_times[i] = lbl_time
                self.lbl_coords[i] = lbl_coord

        # split location and labels to sequences:
        self.max_seq_count = 0           #* maximum number of full--length-20--sequences found for a particular user
        self.min_seq_count = 10_000_000  #* minimum number of full--length-20--sequences found for a particular user
        self.capacity = 0                #* total number of full--length-20--sequences across all users

        with open("./100_users/User-Sequences.txt", "w") as f: 
             # this iterates over users i
            for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(zip(self.times, self.coords, self.locs, self.labels, self.lbl_times, self.lbl_coords)):
                seq_count = len(loc) // sequence_length  #* this is floor; how many full sequence_lengths we have in loc. The following check asserts that loc has at least sequence_length==20 elements
                assert seq_count > 0 , f"fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!; len(loc)={len(loc)}, sequence_length={sequence_length}, len(loc)//sequence_length={len(loc) // sequence_length}; user ID after mapping={i}"  #* DEBUG SET A
                # print(f"len(loc)={len(loc)}, seq_count={seq_count}")
                seqs = []
                seq_times = []
                seq_coords = []
                seq_lbls = []
                seq_lbl_times = []
                seq_lbl_coords = []
                for j in range(seq_count):
                    start = j * sequence_length
                    end = (j+1) * sequence_length
                    seqs.append(loc[start:end])
                    seq_times.append(time[start:end])
                    seq_coords.append(coord[start:end])
                    seq_lbls.append(label[start:end])
                    seq_lbl_times.append(lbl_time[start:end])
                    seq_lbl_coords.append(lbl_coord[start:end])
                self.sequences.append(seqs)
                self.sequences_times.append(seq_times)
                self.sequences_coords.append(seq_coords)
                self.sequences_labels.append(seq_lbls)
                self.sequences_lbl_times.append(seq_lbl_times)
                self.sequences_lbl_coords.append(seq_lbl_coords)
                self.sequences_count.append(seq_count)
                self.capacity += seq_count
                self.max_seq_count = max(self.max_seq_count, seq_count)
                self.min_seq_count = min(self.min_seq_count, seq_count)

                # Write the sequences for each user to the file
                f.write(f"User {i} sequences:\n")
                for j, seq in enumerate(seqs):
                    f.write(f"  Sequence {j}:\n")
                    f.write(f"    Times: {seq_times[j]}\n")
                    f.write(f"    Coords: {seq_coords[j]}\n")
                    f.write(f"    Locs: {seq}\n\n")

    def sequences_by_user(self, idx):
        return self.sequences[idx]

    def __len__(self):
        ''' Amount of available batches to process each sequence at least once.
        '''
        #???????????????????????????????????????????????????????????????????????????????????????????
        # ! For what a batch means, look at __getitem__().
        # ! There is an externally enforced constrain of "batch size must be lower than the amount of available users"
        # I think each batch will select batch_size number of users
        # and potentially ignoring users who do not fill the final batch size (incomplete batches)
        # not considered: (len(self.users) // self.batch_size)

        if (self.usage == Usage.MIN_SEQ_LENGTH):
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            # estimated capacity:
            estimated = self.capacity // self.batch_size
            result = max(self.max_seq_count, estimated)
            # print(f"self.max_seq_count={self.max_seq_count}, self.batch_size={self.batch_size}, result={result}")
            return result
        if (self.usage == Usage.CUSTOM):
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(self, idx):
        ''' Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.
        # ! The external dataloader object using this dataset will, in each "minibatch", call
        # ! __getitem__() only once since batch_size=1 for the dataloader.
        # ! Usually the dataloader will call this batch_size number of times and bunch them together
        # ! and the above happens for each iteration of the dataloader.
        # ! When iterating over the dataloader, a total of dataset __len__() iterations are genreated.
        # ! Read more at https://stackoverflow.com/a/48611864

        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.
        # ! There is an externally enforced constrain of "batch size must be lower than the amount of available users"

        y is the target location and y_t, y_s the targets timestamp and coordiantes. Provided for
        possible use.

        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        '''
        max_iterations = len(self.users)  # Safe upper bound
        iterations = 0

        seqs = []
        times = []
        coords = []
        lbls = []
        lbl_times = []
        lbl_coords = []
        reset_h = []
        for i in range(self.batch_size):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if (self.usage == Usage.MIN_SEQ_LENGTH):
                max_j = self.min_seq_count
            if (self.usage == Usage.CUSTOM):
                max_j = min(max_j, self.custom_seq_count) # use either the users maxima count or limit by custom count
            # print(f"i={i}, j={j}, max_j={max_j}")
            if (j >= max_j):
                # repalce this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # ! Lmao, remember There is an externally enforced constrain of "batch size must be lower than the amount of available users"
                # ! so the while loop below will terminate for sure
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    # raise NotImplementedError # Previously raised NotImplementedError; verified this edge case doesn't occur under current constraints.
                    # print(f"User permutation index: {self.next_user_idx}, Active users: {self.active_users}, Users: {self.users}") # DEBUG
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                    iterations += 1
                    if iterations > max_iterations:
                        raise RuntimeError("Infinite loop detected in user replacement logic.")
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            self.active_user_seq[i] += 1

        #* Each batch represents a full--length-20--sequence from one user (batch_size different users).
        #* Each time __getitem__() is called we pull the next sequence from the user, or replace that user if they have no more sequences...
        #* ... Think of self.active_users as like a bucket/holding area of users.
        x = torch.stack(seqs, dim=1)          #* stack column wise, so each column is a batch (following RNN convention)
        t = torch.stack(times, dim=1)         #* stack column wise, so each column is a batch (following RNN convention)
        s = torch.stack(coords, dim=1)        #* stack column wise, so each column is a batch (following RNN convention)
        y = torch.stack(lbls, dim=1)          #* stack column wise, so each column is a batch (following RNN convention)
        y_t = torch.stack(lbl_times, dim=1)   #* stack column wise, so each column is a batch (following RNN convention)
        y_s = torch.stack(lbl_coords, dim=1)  #* stack column wise, so each column is a batch (following RNN convention)

        return x, t, s, y, y_t, y_s, reset_h, torch.tensor(self.active_users)

