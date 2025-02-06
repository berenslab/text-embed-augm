import datasets
import numpy as np
import torch
from tqdm.notebook import tqdm

from text_embeddings_src.train_stuff import mean_pool,sep_pool,cls_pool
    

@torch.no_grad()
def generate_embeddings(abstracts, tokenizer, model, device, batch_size=256, return_seventh = False):
    """Generate embeddings using BERT-based model.

    Parameters
    ----------
    abstracts : list, this has to be a list not sure if array works but pandas do not work
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".
        
    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    dataset = datasets.Dataset.from_dict(inputs)
    dataset.set_format(type="torch", output_all_columns=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, #num_workers=10
    )

    # new inference
    #model.to(device)

    embedding_av  = []
    embedding_sep = []
    embedding_cls = []
    embedding_7th = []

    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            token_embeds = out[0]  # get the last hidden state
            av = mean_pool(token_embeds, batch["attention_mask"])
            sep = sep_pool(token_embeds, batch["attention_mask"])
            cls = cls_pool(token_embeds, batch["attention_mask"])
            embedding_av.append(av.detach().cpu().numpy())
            embedding_sep.append(sep.detach().cpu().numpy())
            embedding_cls.append(cls.detach().cpu().numpy())
            if return_seventh == True:
                seventh = token_embeds[:, 7, :]
                embedding_7th.append(seventh.detach().cpu().numpy())
    
    
    embedding_av = np.vstack(embedding_av)
    embedding_sep = np.vstack(embedding_sep)
    embedding_cls = np.vstack(embedding_cls)
    
    if return_seventh == True:
        embedding_7th = np.vstack(embedding_7th)

    return (embedding_cls, embedding_sep, embedding_av, embedding_7th) if return_seventh == True else (embedding_cls, embedding_sep, embedding_av)



@torch.no_grad()
def generate_embeddings_embed_layer(
    abstracts, tokenizer, model, device, batch_size=256, return_seventh=False
):
    """Generate embeddings using BERT-based model.
    # valid function for both the layer and the module (04/09/2024)

    Parameters
    ----------
    abstracts : list, this has to be a list not sure if array works but pandas do not work
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".

    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    dataset = datasets.Dataset.from_dict(inputs)
    dataset.set_format(type="torch", output_all_columns=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # num_workers=10
    )

    # new inference
    # model.to(device)

    embedding_av = []
    embedding_sep = []
    embedding_cls = []
    embedding_7th = []

    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"])
            token_embeds = out  # [0]  # get the last hidden state
            av = mean_pool(token_embeds, batch["attention_mask"])
            sep = sep_pool(token_embeds, batch["attention_mask"])
            cls = cls_pool(token_embeds, batch["attention_mask"])
            embedding_av.append(av.detach().cpu().numpy())
            embedding_sep.append(sep.detach().cpu().numpy())
            embedding_cls.append(cls.detach().cpu().numpy())
            if return_seventh == True:
                seventh = token_embeds[:, 7, :]
                embedding_7th.append(seventh.detach().cpu().numpy())

    embedding_av = np.vstack(embedding_av)
    embedding_sep = np.vstack(embedding_sep)
    embedding_cls = np.vstack(embedding_cls)

    if return_seventh == True:
        embedding_7th = np.vstack(embedding_7th)

    return (
        (embedding_cls, embedding_sep, embedding_av, embedding_7th)
        if return_seventh == True
        else (embedding_cls, embedding_sep, embedding_av)
    )



# TODO: this function has not been adapted to the new code
@torch.no_grad()
def generate_embeddings_hidden_state(
    layer_number,
    abstracts,
    tokenizer,
    model,
    device,
    batch_size=2048,
    return_seventh=False,
):
    """Generate embeddings using BERT-based model.

    Parameters
    ----------
    abstracts : list, this has to be a list not sure if array works but pandas do not work
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".

    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    dataset = datasets.Dataset.from_dict(inputs)
    dataset.set_format(type="torch", output_all_columns=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=10
    )

    embedding_av = []
    embedding_sep = []
    embedding_cls = []
    embedding_7th = []

    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True)
            token_embeds = out.hidden_states[layer_number]
            av = mean_pool(token_embeds, batch["attention_mask"])
            sep = sep_pool(token_embeds, batch["attention_mask"])
            cls = cls_pool(token_embeds, batch["attention_mask"])
            embedding_av.append(av.detach().cpu().numpy())
            embedding_sep.append(sep.detach().cpu().numpy())
            embedding_cls.append(cls.detach().cpu().numpy())
            if return_seventh == True:
                seventh = token_embeds[:, 7, :]
                embedding_7th.append(seventh.detach().cpu().numpy())

    embedding_av = np.vstack(embedding_av)
    embedding_sep = np.vstack(embedding_sep)
    embedding_cls = np.vstack(embedding_cls)

    if return_seventh == True:
        embedding_7th = np.vstack(embedding_7th)

    return (
        (embedding_cls, embedding_sep, embedding_av, embedding_7th)
        if return_seventh == True
        else (embedding_cls, embedding_sep, embedding_av)
    )


