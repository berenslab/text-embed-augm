import itertools
import re

import numpy as np
import torch
from transformers import AutoTokenizer


# def return_pair(abstract, rng=None):
#     sentences = abstract.split(".")
#     sentences = [s.strip() + "." for s in sentences if len(s) >= 100]

#     rng = np.random if rng is None else rng
#     if len(sentences) >= 2:
#         idx = rng.choice(len(sentences), size=2, replace=False)
#         return np.array(sentences)[idx]
#     else:
#         return []


# def get_labeled_iclr(iclr):
#     keywords = [
#         "network",
#         "graph",
#         "reinforcement",
#         "language",
#         "adversarial",
#         "federated",
#         "contrastive",
#         "domain",
#         "diffusion",
#         "out-of-dis",
#         "continual",
#         "distillation",
#         "architecture",
#         "privacy",
#         "protein",
#         "fair",
#         "attention",
#         "video",
#         "meta-learning",
#         "generative adv",
#         "autoencoder",
#         "game",
#         "semi-sup",
#         "pruning",
#         "physics",
#         "3d",
#         "translation",
#         "optimization",
#         "recurrent",
#         "word",
#         "bayesian",
#     ]
#     keywords = np.array(keywords)

#     y = np.zeros(iclr.shape[0]) * np.nan

#     for num, keyword in enumerate(keywords):
#         mask = [keyword.lower() in t.lower() for t in iclr.title]
#         y[mask & ~np.isnan(y)] = -1
#         y[mask & np.isnan(y)] = num

#     # print(y.size)
#     # print(np.sum(~np.isnan(y)))
#     # print(np.sum(y >= 0))

#     labeled = y >= 0

#     iclr_labeled = iclr[labeled].reset_index(drop=True)
#     y_labeled = y[labeled].astype(int)
#     iclr_labeled["y"] = y_labeled
#     iclr_labeled["label"] = keywords[y_labeled]
#     return iclr_labeled


# def get_tokenizer(pretrained_name="allenai/scibert_scivocab_uncased"):
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
#     return tokenizer


class SentencePairDataset(torch.utils.data.Dataset):
    def __init__(self, abstracts, tokenizer, device, tokenizer_kwargs=None, seed=42):
        # actually a list of tokens
        self.abstracts = abstracts
        self.rng = np.random.default_rng(seed)

        self.sentences_map = [
            (s.strip() + ".", i)
            for i, sentences in enumerate(
                abstracts.map(lambda a: a.split("."))
            )
            for s in sentences
            if (len(s) >= 100) & (len(s) <= 250)  #limit the sentence lengths
        ]

        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        self.sentences_tok = tokenizer(
            [x for x, _ in self.sentences_map], **tokenizer_kwargs
        ).to(device)


        # we group the flat sentences by the original abstract they
        # come from.  Then we can check whether we have enough
        # sentences and append the abstracts with at least two
        # sentences to our list.
        sentences_and_toks = zip(
            self.sentences_map,
            self.sentences_tok["input_ids"],
            self.sentences_tok["attention_mask"],
        )
        self.abs_sentences = []
        self.abs_toks = []
        self.abs_amsk = []
        for key, group in itertools.groupby(
            sentences_and_toks,
            key=lambda kvtoksetc: kvtoksetc[0][1],
        ):
            grp = list(group)
            if len(grp) < 2:
                continue  # not enough sentences
            else:
                self.abs_sentences.append([kv[0] for kv in grp])
                self.abs_toks.append([x[1] for x in grp])
                self.abs_amsk.append([x[2] for x in grp])

        # we now have `self.abs_toks`, which is a list of lists,
        # where the first list is the abstracts and the second is the
        # token representation of the sentences within the given
        # abstract.

    def __getitem__(self, idx):
        """
        In this function you input a index idx and it selects from that abstract two random sentences.
        Because it is the getitem method, it means that when you index sth it will automatically return you this two random sentences from that particular abstract.
        """
        abstract = self.abs_toks[idx]
        amask = self.abs_amsk[idx]
        i1, i2 = self.rng.choice(len(abstract), size=2, replace=False)
        return (abstract[i1], amask[i1]), (abstract[i2], amask[i2])

    def __len__(self):
        return len(self.abs_sentences)


class MultSentencesPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        abstracts,
        tokenizer,
        device,
        n_cons_sntcs=2,
        tokenizer_kwargs=None,
        seed=42,
    ):
        # actually a list of tokens
        self.abstracts = abstracts
        self.rng = np.random.default_rng(seed)
        self.n_cons_sntcs=n_cons_sntcs

        regex_block = r".{5,}?\."
        _regex = regex_block
        for i in range(self.n_cons_sntcs - 1):
            _regex += r"\s" + regex_block

        self.sentences_map = [
            (s.strip() + ".", i)
            for i, sentences in enumerate(
                abstracts.map(lambda a: re.findall(_regex, a, flags=re.S))
            )
            for s in sentences
            if (len(s) >= 100*self.n_cons_sntcs) & (len(s) <= 250*self.n_cons_sntcs)  #limit the sentence lengths
        ]

        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        self.sentences_tok = tokenizer(
            [x for x, _ in self.sentences_map], **tokenizer_kwargs
        ).to(device)


        # we group the flat sentences by the original abstract they
        # come from.  Then we can check whether we have enough
        # sentences and append the abstracts with at least two
        # sentences to our list.
        sentences_and_toks = zip(
            self.sentences_map,
            self.sentences_tok["input_ids"],
            self.sentences_tok["attention_mask"],
        )
        self.abs_sentences = []
        self.abs_toks = []
        self.abs_amsk = []
        for key, group in itertools.groupby(
            sentences_and_toks,
            key=lambda kvtoksetc: kvtoksetc[0][1],
        ):
            grp = list(group)
            if len(grp) < 2:
                continue  # not enough sentences
            else:
                self.abs_sentences.append([kv[0] for kv in grp])
                self.abs_toks.append([x[1] for x in grp])
                self.abs_amsk.append([x[2] for x in grp])

        # we now have `self.abs_toks`, which is a list of lists,
        # where the first list is the abstracts and the second is the
        # token representation of the sentences within the given
        # abstract.

    def __getitem__(self, idx):
        abstract = self.abs_toks[idx]
        amask = self.abs_amsk[idx]
        i1, i2 = self.rng.choice(len(abstract), size=2, replace=False)
        return (abstract[i1], amask[i1]), (abstract[i2], amask[i2])

    def __len__(self):
        return len(self.abs_sentences)


class MultOverlappingSentencesPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        abstracts,
        tokenizer,
        device,
        n_cons_sntcs=2,
        tokenizer_kwargs=None,
        seed=42,
    ):
        # actually a list of tokens
        self.abstracts = abstracts
        self.rng = np.random.default_rng(seed)
        self.n_cons_sntcs=n_cons_sntcs
        
        # sentence map
        self.sentences_map = []
        for i, sentences in enumerate(
            abstracts.map(lambda a: a.split("."))
        ):  # loop through abstracts
            for j in range(
                len(sentences) - (self.n_cons_sntcs - 1)
            ):  # loop through sentences inside abstract
                if (len(sentences[j]) >= 100) & (len(sentences[j]) <= 250):  # length conditions
                    cons_sentences_pack = ""
                    cons_sentence_counts = 0
                    for k in range(
                        len(sentences) - j
                    ):  # loop through sentences to add them
                        if (len(sentences[j + k]) >= 100) & (len(sentences[j+k]) <= 250):  # length conditions
                            cons_sentences_pack += sentences[j + k].strip() + ". "
                            cons_sentence_counts += 1

                        if (
                            cons_sentence_counts == self.n_cons_sntcs
                        ):  # check if we have already enough sentences
                            self.sentences_map.append((cons_sentences_pack, i))
                            break


        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        self.sentences_tok = tokenizer(
            [x for x, _ in self.sentences_map], **tokenizer_kwargs
        ).to(device)


        # we group the flat sentences by the original abstract they
        # come from.  Then we can check whether we have enough
        # sentences and append the abstracts with at least two
        # sentences to our list.
        sentences_and_toks = zip(
            self.sentences_map,
            self.sentences_tok["input_ids"],
            self.sentences_tok["attention_mask"],
        )
        self.abs_sentences = []
        self.abs_toks = []
        self.abs_amsk = []
        for key, group in itertools.groupby(
            sentences_and_toks,
            key=lambda kvtoksetc: kvtoksetc[0][1],
        ):
            grp = list(group)
            if len(grp) < 2:
                continue  # not enough sentences
            else:
                self.abs_sentences.append([kv[0] for kv in grp])
                self.abs_toks.append([x[1] for x in grp])
                self.abs_amsk.append([x[2] for x in grp])

        # we now have `self.abs_toks`, which is a list of lists,
        # where the first list is the abstracts and the second is the
        # token representation of the sentences within the given
        # abstract.

    def __getitem__(self, idx):
        abstract = self.abs_toks[idx]
        amask = self.abs_amsk[idx]
        i1, i2 = self.rng.choice(len(abstract), size=2, replace=False)
        return (abstract[i1], amask[i1]), (abstract[i2], amask[i2])

    def __len__(self):
        return len(self.abs_sentences)


class MultOverlappingSentencesLabelPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        abstracts,
        labels,
        tokenizer,
        device,
        n_cons_sntcs=2,
        tokenizer_kwargs=None,
        seed=42,
    ):
        # actually a list of tokens
        self.abstracts = abstracts
        self.labels = labels
        self.n_cons_sntcs = n_cons_sntcs
        self.rng = np.random.default_rng(seed)

        # sentence map
        self.sentences_map = []
        for i, sentences in enumerate(
            abstracts.map(lambda a: a.split("."))
        ):  # loop through abstracts
            for j in range(
                len(sentences) - (self.n_cons_sntcs - 1)
            ):  # loop through sentences inside abstract
                if (len(sentences[j]) >= 100) & (
                    len(sentences[j]) <= 250
                ):  # length conditions
                    cons_sentences_pack = ""
                    cons_sentence_counts = 0
                    for k in range(
                        len(sentences) - j
                    ):  # loop through sentences to add them
                        if (len(sentences[j + k]) >= 100) & (
                            len(sentences[j + k]) <= 250
                        ):  # length conditions
                            cons_sentences_pack += (
                                sentences[j + k].strip() + ". "
                            )
                            cons_sentence_counts += 1

                        if (
                            cons_sentence_counts == self.n_cons_sntcs
                        ):  # check if we have already enough sentences
                            self.sentences_map.append(
                                (cons_sentences_pack, i, self.labels[i])
                            )
                            break

        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        self.sentences_tok = tokenizer(
            [x for x, _, _ in self.sentences_map], **tokenizer_kwargs
        ).to(device)

        # we group the flat sentences by the original abstract they
        # come from.  Then we can check whether we have enough
        # sentences and append the abstracts with at least two
        # sentences to our list.
        sentences_and_toks = zip(
            self.sentences_map,
            self.sentences_tok["input_ids"],
            self.sentences_tok["attention_mask"],
        )
        self.abs_sentences = []
        self.abs_toks = []
        self.abs_amsk = []
        for key, group in itertools.groupby(
            sentences_and_toks,
            key=lambda kvtoksetc: kvtoksetc[0][1],
        ):
            grp = list(group)
            if len(grp) < 2:
                continue  # not enough sentences
            else:
                self.abs_sentences.append([kv[0] for kv in grp])
                self.abs_toks.append([x[1] for x in grp])
                self.abs_amsk.append([x[2] for x in grp])

        # we now have `self.abs_toks`, which is a list of lists,
        # where the first list is the abstracts and the second is the
        # token representation of the sentences within the given
        # abstract.

    def __getitem__(self, idx):
        # select sentence 1
        isentence1 = self.rng.choice(len(self.abs_sentences[idx]), replace=False)
        abstract1_info = self.abs_sentences[idx][isentence1]
        output1 = (
            self.abs_toks[idx][isentence1],
            self.abs_amsk[idx][isentence1],
        )

        # select sentence 2
        abstract1_label = abstract1_info[2]
        abstract1_idx = abstract1_info[1]

        abstracts_same_label = [
            elem
            for i, elem in enumerate(
                zip(self.abs_sentences, self.abs_toks, self.abs_amsk)
            )
            if (elem[0][0][2] == abstract1_label)
            & (elem[0][0][1] != abstract1_idx)
        ]

        iabstract2 = self.rng.choice(len(abstracts_same_label), replace=False)
        isentence2 = self.rng.choice(
            len(abstracts_same_label[iabstract2][0]), replace=False
        )

        output2 = (
            abstracts_same_label[iabstract2][1][isentence2],
            abstracts_same_label[iabstract2][2][isentence2],
        )
        return output1, output2

    def __len__(self):
        return len(self.abs_sentences)


class AbstractSplitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        abstracts,
        tokenizer,
        tokenizer_kwargs=None,
    ):
        # actually a list of tokens
        self.abstracts = abstracts

        self.sentences_map = dict()
        regex_block = r".{5,}?\."
        for i, (abstract) in enumerate(abstracts):
            n_sntcs = sum(
                1
                for _ in re.finditer(regex_block + r"\s", abstract, flags=re.S)
            )
            # n_sntcs = abstract.count(".")

            _regex = regex_block + "".join(
                [r"\s" + regex_block for _ in range(n_sntcs // 2)]
            )

            part1 = re.match(_regex, abstract, flags=re.S)[0]
            part2 = re.sub(_regex, "", abstract, flags=re.S)
            self.sentences_map[part1] = i
            self.sentences_map[part2] = i

        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        self.sentences_tok = tokenizer(
            list(self.sentences_map.keys()), **tokenizer_kwargs
        )

        # we group the flat sentences by the original abstract they
        # come from.  Then we can check whether we have enough
        # sentences and append the abstracts with at least two
        # sentences to our list.
        sentences_and_toks = zip(
            self.sentences_map.items(),
            self.sentences_tok["input_ids"],
            self.sentences_tok["attention_mask"],
        )
        self.abs_sentences = []
        self.abs_toks = []
        self.abs_amsk = []
        for key, group in itertools.groupby(
            sentences_and_toks,
            key=lambda kvtoksetc: kvtoksetc[0][1],
        ):
            grp = list(group)
            if len(grp) < 2:
                continue  # not enough sentences
            else:
                self.abs_sentences.append([kv[0] for kv in grp])
                self.abs_toks.append([x[1] for x in grp])
                self.abs_amsk.append([x[2] for x in grp])

        # we now have `self.abs_toks`, which is a list of lists,
        # where the first list is the abstracts and the second is the
        # token representation of the sentences within the given
        # abstract.

    def __getitem__(self, idx):
        abstract = self.abs_toks[idx]
        amask = self.abs_amsk[idx]
        i1, i2 = 0, 1
        return (abstract[i1], amask[i1]), (abstract[i2], amask[i2])

    def __len__(self):
        return len(self.abs_sentences)


class MaskedAbstractDataset(torch.utils.data.Dataset):
    """FRACTION OF THE INDIVIDUAL ABSTRACT LENGTH"""

    def __init__(
        self,
        abstracts,
        tokenizer,
        device,
        tokenizer_kwargs=None,
        fraction_masked=0.4,
        truncate = False,
        cut_off = 1500,
        seed=42,
    ):
        # actually a list of tokens
        self.abstracts = abstracts
        self.rng = np.random.default_rng(seed)
        self.fraction_masked = fraction_masked
        self.device=device
        self.cut_off = cut_off

        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,  # this one is new and important for this class
            )

        if truncate == True:
            self.abstracts = [elem[:self.cut_off] for elem in self.abstracts]

        self.abstracts_tok = tokenizer(self.abstracts, **tokenizer_kwargs).to(
            self.device
        )

    def __getitem__(self, idx):
        """
        In this function you input a index idx and it selects ...
        """
        # get original abstracts and masks
        abstract = self.abstracts_tok["input_ids"][idx]
        amask = self.abstracts_tok["attention_mask"][idx]
        special_tokens_mask = self.abstracts_tok["special_tokens_mask"][idx]

        masked_token_value = (
            torch.ones(abstract.size(), dtype=int, device=self.device) * 103
        )
        real_abstract_length = int((special_tokens_mask == 0).sum())

        # masked abstract 1 -- maybe make this into function; ask Nik
        mask_1 = np.zeros(real_abstract_length, dtype=int)
        mask_1[: round(real_abstract_length * self.fraction_masked)] = 1
        self.rng.shuffle(mask_1)
        mask_1 = np.pad(
            mask_1, (1, len(abstract) - real_abstract_length - 1)
        )  # pad 1 zero at the beggining for the CLS token, and pad the end to fill until reaching original length
        mask_1 = torch.from_numpy(mask_1).to(self.device)

        abstract_1 = torch.where(
            (special_tokens_mask == 0) & (mask_1 == 1),
            masked_token_value,
            abstract,
        )

        # masked abstract 2 -- maybe make this into function; ask Nik
        mask_2 = np.zeros(real_abstract_length, dtype=int)
        mask_2[: round(real_abstract_length * self.fraction_masked)] = 1
        self.rng.shuffle(mask_2)
        mask_2 = np.pad(mask_2, (1, len(abstract) - real_abstract_length - 1))
        mask_2 = torch.from_numpy(mask_2).to(self.device)

        abstract_2 = torch.where(
            (special_tokens_mask == 0) & (mask_2 == 1),
            masked_token_value,
            abstract,
        )

        return (abstract_1, amask), (abstract_2, amask)

    def __len__(self):
        return len(self.abstracts)
    


class MaskedMultOverlappingSentencesPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        abstracts,
        tokenizer,
        device,
        n_cons_sntcs=2,
        tokenizer_kwargs=None,
        fraction_masked=0.4,
        seed=42,
    ):
        # actually a list of tokens
        self.abstracts = abstracts
        self.rng = np.random.default_rng(seed)
        self.n_cons_sntcs = n_cons_sntcs
        self.device = device
        self.fraction_masked = fraction_masked

        # sentence map
        self.sentences_map = []
        for i, sentences in enumerate(
            abstracts.map(lambda a: a.split("."))
        ):  # loop through abstracts
            for j in range(
                len(sentences) - (self.n_cons_sntcs - 1)
            ):  # loop through sentences inside abstract
                if (len(sentences[j]) >= 100) & (
                    len(sentences[j]) <= 250
                ):  # length conditions
                    cons_sentences_pack = ""
                    cons_sentence_counts = 0
                    for k in range(
                        len(sentences) - j
                    ):  # loop through sentences to add them
                        if (len(sentences[j + k]) >= 100) & (
                            len(sentences[j + k]) <= 250
                        ):  # length conditions
                            cons_sentences_pack += (
                                sentences[j + k].strip() + ". "
                            )
                            cons_sentence_counts += 1

                        if (
                            cons_sentence_counts == self.n_cons_sntcs
                        ):  # check if we have already enough sentences
                            self.sentences_map.append((cons_sentences_pack, i))
                            break

        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict(
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,  # this one is new and important for this class
            )

        self.sentences_tok = tokenizer(
            [x for x, _ in self.sentences_map], **tokenizer_kwargs
        ).to(self.device)

        # we group the flat sentences by the original abstract they
        # come from.  Then we can check whether we have enough
        # sentences and append the abstracts with at least two
        # sentences to our list.
        sentences_and_toks = zip(
            self.sentences_map,
            self.sentences_tok["input_ids"],
            self.sentences_tok["attention_mask"],
            self.sentences_tok["special_tokens_mask"],
        )
        self.abs_sentences = []
        self.abs_toks = []
        self.abs_mask = []
        self.abs_special_msk = []
        for key, group in itertools.groupby(
            sentences_and_toks,
            key=lambda kvtoksetc: kvtoksetc[0][
                1
            ],  # this is selecting the index from the abstract from the tuple inside sentences_map
        ):
            grp = list(group)
            if len(grp) < 2:
                continue  # not enough sentences
            else:
                self.abs_sentences.append([kv[0] for kv in grp])
                self.abs_toks.append([x[1] for x in grp])
                self.abs_mask.append([x[2] for x in grp])
                self.abs_special_msk.append([x[3] for x in grp])

        # we now have `self.abs_toks`, which is a list of lists,
        # where the first list is the abstracts and the second is the
        # token representation of the sentences within the given
        # abstract.

    def __getitem__(self, idx):
        abstract = self.abs_toks[idx]
        amask = self.abs_mask[idx]
        special_tokens_mask = self.abs_special_msk[idx]
        i1, i2 = self.rng.choice(len(abstract), size=2, replace=False)

        # select pair of cons sentences
        sentences_1 = abstract[i1]
        amask_1 = amask[i1]
        special_tokens_mask_1 = special_tokens_mask[i1]
        sentences_2 = abstract[i2]
        amask_2 = amask[i2]
        special_tokens_mask_2 = special_tokens_mask[i2]

        # sentences 1
        masked_token_value_1 = (
            torch.ones(sentences_1.size(), dtype=int, device=self.device) * 103
        )
        real_sentences_length_1 = int((special_tokens_mask_1 == 0).sum())

        # masked sentences 1 -- maybe make this into function; ask Nik
        mask_1 = np.zeros(real_sentences_length_1, dtype=int)
        mask_1[: round(real_sentences_length_1 * self.fraction_masked)] = 1
        self.rng.shuffle(mask_1)
        mask_1 = np.pad(
            mask_1, (1, len(sentences_1) - real_sentences_length_1 - 1)
        )  # pad 1 zero at the beggining for the CLS token, and pad the end to fill until reaching original length
        mask_1 = torch.from_numpy(mask_1).to(self.device)

        sentences_1_masked = torch.where(
            (special_tokens_mask_1 == 0) & (mask_1 == 1),
            masked_token_value_1,
            sentences_1,
        )

        # sentences 2
        masked_token_value_2 = (
            torch.ones(sentences_2.size(), dtype=int, device=self.device) * 103
        )
        real_sentences_length_2 = int((special_tokens_mask_2 == 0).sum())

        # masked sentences 2 -- maybe make this into function; ask Nik
        mask_2 = np.zeros(real_sentences_length_2, dtype=int)
        mask_2[: round(real_sentences_length_2 * self.fraction_masked)] = 1
        self.rng.shuffle(mask_2)
        mask_2 = np.pad(
            mask_2, (1, len(sentences_2) - real_sentences_length_2 - 1)
        )
        mask_2 = torch.from_numpy(mask_2).to(self.device)

        sentences_2_masked = torch.where(
            (special_tokens_mask_2 == 0) & (mask_2 == 1),
            masked_token_value_2,
            sentences_2,
        )

        return (sentences_1_masked, amask_1), (sentences_2_masked, amask_2)

    def __len__(self):
        return len(self.abs_sentences)