#import datasets
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm
from transformers import AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from text_embeddings_src.metrics import knn_accuracy, knn_recall, logistic_accuracy
from text_embeddings_src.embeddings import generate_embeddings
from text_embeddings_src.dim_red import run_tsne_simple


def fix_all_seeds(seed=42):
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  ## this one is new
    ## Set the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) ## this one I don't use but don't remember why

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed
    random.seed(seed)


# pooling functions
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size())
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def sep_pool(token_embeds, attention_mask):
    ix = attention_mask.sum(1) - 1
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, ix, :]

def cls_pool(token_embeds, attention_mask):
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, 0, :]

def seventh_pool(token_embeds, attention_mask):
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, 7, :]



#def get_model(pretrained_str="allenai/scibert_scivocab_uncased"):
#    model = AutoModel.from_pretrained(pretrained_str)
#    return model


def train_loop(
    model,
    loader,
    device,
    titles_abstracts_together,
    tokenizer,
    label_mask,
    labels_acc,
    optimized_rep="av",
    n_epochs=1,
    lr=2e-5,
    eval_metric= "knn",
    return_seventh=False,
    return_embeddings= False,
    return_model= False,
):
    assert optimized_rep in [
        "av",
        "cls",
        "sep",
        "7th",
    ], "Not valid `optimized_rep`. Choose from ['av', 'cls', 'sep', '7th']."

    assert eval_metric in [
        "knn",
        "lin",
    ], "Not valid `eval_metric`. Choose from ['knn', 'lin']."

    model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value, it is the inverse of the temperature
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs 
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))
    accuracies = []
    for epoch in range(n_epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(device)     # this are all anchor abstracts from the batch,len(anchor_ids)= len(batch)
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)       # this each positive pair from each anchor, all in one array, also len(batch)
            pos_mask = batch[1][1].to(device)
            # extract token embeddings from BERT
            a = model(anchor_ids, attention_mask=anchor_mask)[0]  # all token embeddings
            p = model(pos_ids, attention_mask=pos_mask)[0]
            
            # get the mean pooled vectors  -- put all of these ifs into a pool function (wraper) to which I pass, a, p the masks and the optimized rep
            if optimized_rep == "av":
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                
            elif optimized_rep == "cls":
                a = cls_pool(a, anchor_mask)
                p = cls_pool(p, pos_mask)
                
            elif optimized_rep == "sep":
                a = sep_pool(a, anchor_mask)
                p = sep_pool(p, pos_mask)
                
            elif optimized_rep == "7th":
                a = seventh_pool(a, anchor_mask)
                p = seventh_pool(p, pos_mask)
                
            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )  # I think that the labels are just the label of which pair it is. 0 for the first pair, 1 for the second...
            # my guess is that they are used in the loss to know which of the cosine similarities should be high
            # and which low

            # and now calculate the loss
            loss = loss_func(scores * scale, labels) 
            losses[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            
        ## evaluation -- externalize all of these code into an "evaluation" fucntion
        if return_seventh == True:
            (
                embedding_cls,
                embedding_sep,
                embedding_av,
                embedding_7th,
            ) = generate_embeddings(
                titles_abstracts_together,
                tokenizer,
                model,
                device,
                batch_size=256,
                return_seventh=True,
            )
            if eval_metric == "knn":
                acc = knn_accuracy(
                    [
                        embedding_av[label_mask],
                        embedding_cls[label_mask],
                        embedding_sep[label_mask],
                        embedding_7th[label_mask],
                    ],
                    labels_acc,
                )
            elif eval_metric == "lin":
                acc = logistic_accuracy([embedding_av[label_mask], embedding_cls[label_mask], embedding_sep[label_mask], embedding_7th[label_mask]], labels_acc)
    

        else:
            (
                embedding_cls,
                embedding_sep,
                embedding_av,
            ) = generate_embeddings(
                titles_abstracts_together,
                tokenizer,
                model,
                device,
                batch_size=256,
            )
            if eval_metric == "knn":
                acc = knn_accuracy(
                    [
                        embedding_av[label_mask],
                        embedding_cls[label_mask],
                        embedding_sep[label_mask],
                    ],
                    labels_acc,
                )
            elif eval_metric == "lin":
                acc = logistic_accuracy([embedding_av[label_mask], embedding_cls[label_mask], embedding_sep[label_mask]], labels_acc)


        accuracies.append(acc)
        
    # returns
    if return_embeddings==True:
        if return_seventh == True:
            return losses, accuracies, embedding_cls, embedding_sep, embedding_av, embedding_7th
        if return_seventh == False:
            return losses, accuracies, embedding_cls, embedding_sep, embedding_av
    elif return_model == True:
        return losses, accuracies, model
    else:
        return losses, accuracies


#def embed_tokens(model, tokens, batch_size=2048):
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model.to(device)
#
#    dataset = datasets.Dataset.from_dict(tokens)
#    dataset.set_format(type="torch", output_all_columns=True)
#    loader = torch.utils.data.DataLoader(
#        dataset, batch_size=batch_size, num_workers=32
#    )
#
#    Ys = []
#    with torch.no_grad():
#        model.eval()
#        for batch in tqdm(loader):
#            batch = {k: v.to(device) for k, v in batch.items()}
#            out = model(**batch)
#            embs = out[0]  # get the last hidden state
#            Y = mean_pool(embs, batch["attention_mask"])
#            Ys.append(Y.detach().cpu().numpy())
#    print(Ys[0].shape, Ys[-1].shape)
#    return np.vstack(Ys)

def train_loop_batches_eval(
    model,
    loader,
    device,
    titles_abstracts_together,
    tokenizer,
    label_mask,
    labels_acc,
    optimized_rep="av",
    n_epochs=1,
    lr=2e-5,
    eval_metric= "knn",
    return_seventh=False,
):
    assert optimized_rep in [
        "av",
        "cls",
        "sep",
        "7th",
    ], "Not valid `optimized_rep`. Choose from ['av', 'cls', 'sep', '7th']."

    assert eval_metric in [
        "knn",
        "lin",
    ], "Not valid `eval_metric`. Choose from ['knn', 'lin']."

    


    model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))

    accuracies = []
    for epoch in range(n_epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and more all to the active device
            anchor_ids = batch[0][0].to(device)
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)
            pos_mask = batch[1][1].to(device)
            # extract token embeddings from BERT
            a = model(anchor_ids, attention_mask=anchor_mask)[
                0
            ]  # all token embeddings
            p = model(pos_ids, attention_mask=pos_mask)[0]

            # get the mean pooled vectors
            if optimized_rep == "av":
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)

            elif optimized_rep == "cls":
                a = cls_pool(a, anchor_mask)
                p = cls_pool(p, pos_mask)

            elif optimized_rep == "sep":
                a = sep_pool(a, anchor_mask)
                p = sep_pool(p, pos_mask)

            elif optimized_rep == "7th":
                a = seventh_pool(a, anchor_mask)
                p = seventh_pool(p, pos_mask)

            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )
            # and now calculate the loss
            loss = loss_func(
                scores * scale, labels
            )  # Nik does not know what the labels nor the scale are
            losses[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

            ## evaluation
            if (i_batch%10 == 0) | (i_batch == len(loader) - 1):
                if return_seventh == True:
                    (
                        embedding_cls,
                        embedding_sep,
                        embedding_av,
                        embedding_7th,
                    ) = generate_embeddings(
                        titles_abstracts_together,
                        tokenizer,
                        model,
                        device,
                        batch_size=256,
                        return_seventh=True,
                    )
                    if eval_metric == "knn":
                        acc = knn_accuracy(
                            [
                                embedding_av[label_mask],
                                embedding_cls[label_mask],
                                embedding_sep[label_mask],
                                embedding_7th[label_mask],
                            ],
                            labels_acc,
                        )
                    elif eval_metric == "lin":
                        acc = logistic_accuracy([embedding_av[label_mask], embedding_cls[label_mask], embedding_sep[label_mask], embedding_7th[label_mask]], labels_acc)
            

                else:
                    (
                        embedding_cls,
                        embedding_sep,
                        embedding_av,
                    ) = generate_embeddings(
                        titles_abstracts_together,
                        tokenizer,
                        model,
                        device,
                        batch_size=256,
                    )
                    if eval_metric == "knn":
                        acc = knn_accuracy(
                            [
                                embedding_av[label_mask],
                                embedding_cls[label_mask],
                                embedding_sep[label_mask],
                            ],
                            labels_acc,
                        )
                    elif eval_metric == "lin":
                        acc = logistic_accuracy([embedding_av[label_mask], embedding_cls[label_mask], embedding_sep[label_mask]], labels_acc)


                accuracies.append(acc)

    return losses, accuracies, model



def train_loop_tsne_and_knn_rec(model, loader, device, titles_abstracts_together, tokenizer, optimized_rep= "av", n_epochs=1, lr=2e-5):

    assert optimized_rep in ["av", "cls", "sep", "7th"], "Not valid `optimized_rep`. Choose from ['av', 'cls', 'sep', '7th']."

    model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs 
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))
    high_d_reps= []
    tsne_embeddings= []
    
    for epoch in range(n_epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and more all to the active device
            anchor_ids = batch[0][0].to(device)     
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)
            pos_mask = batch[1][1].to(device)
            # extract token embeddings from BERT
            a = model(anchor_ids, attention_mask=anchor_mask)[0]  # all token embeddings
            p = model(pos_ids, attention_mask=pos_mask)[0]
            
            # get the mean pooled vectors
            if optimized_rep == "av":
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                
            elif optimized_rep == "cls":
                a = cls_pool(a, anchor_mask)
                p = cls_pool(p, pos_mask)
                
            elif optimized_rep == "sep":
                a = sep_pool(a, anchor_mask)
                p = sep_pool(p, pos_mask)
                
            elif optimized_rep == "7th":
                a = seventh_pool(a, anchor_mask)
                p = seventh_pool(p, pos_mask)
                
            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )
            # and now calculate the loss
            loss = loss_func(scores * scale, labels)   # Nik does not know what the labels nor the scale are
            losses[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            
        ## get high-dim and low-dim representations
        if (epoch == 0) | (epoch == n_epochs-1):

            embedding_cls, embedding_sep, embedding_av = generate_embeddings(
                titles_abstracts_together, tokenizer, model, device, batch_size=256, return_seventh=False
            )

            if optimized_rep == "av":
                high_d_reps.append(embedding_av)
                tsne_result = run_tsne_simple(embedding_av)
                tsne_embeddings.append(tsne_result)
                
            elif optimized_rep == "cls":
                high_d_reps.append(embedding_cls)
                tsne_result = run_tsne_simple(embedding_cls)
                tsne_embeddings.append(tsne_result)
                
            elif optimized_rep == "sep":
                high_d_reps.append(embedding_sep)
                tsne_result = run_tsne_simple(embedding_sep)
                tsne_embeddings.append(tsne_result)


    # knn recall
    knn_recall_result = knn_recall(high_d_reps[1], high_d_reps[0])
        
    return losses, tsne_embeddings, knn_recall_result



def train_loop_eval_test_loss(model, train_loader, test_loader, device, tokenizer, train_dataset, test_dataset, optimized_rep= "av", n_epochs=1, lr=2e-5):
    
    assert optimized_rep in ["av", "cls", "sep", "7th"], "Not valid `optimized_rep`. Choose from ['av', 'cls', 'sep', '7th']."

    model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(train_loader) * n_epochs 
    warmup_steps = int(0.1 * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses_train = np.empty((n_epochs, len(train_loader)))
    losses_test = np.empty((n_epochs, len(test_loader)))
    knn_accuracies_train =  np.empty((n_epochs, 3))
    knn_accuracies_test =  np.empty((n_epochs, 3))
    for epoch in range(n_epochs):
        model.train()  # make sure model is in training mode

        # TRAIN LOADER -- training model with train loader
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        train_loop = tqdm(train_loader, leave=True)
        for i_batch, batch in enumerate(train_loop):
            # zero all gradients on each new step
            optim.zero_grad()

            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(device)     
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)
            pos_mask = batch[1][1].to(device)
            # extract token embeddings from model
            a = model(anchor_ids, attention_mask=anchor_mask)[0]  # all token embeddings
            p = model(pos_ids, attention_mask=pos_mask)[0]
            
            # get the pooled vectors
            if optimized_rep == "av":
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                
            elif optimized_rep == "cls":
                a = cls_pool(a, anchor_mask)
                p = cls_pool(p, pos_mask)
                
            elif optimized_rep == "sep":
                a = sep_pool(a, anchor_mask)
                p = sep_pool(p, pos_mask)
                
            elif optimized_rep == "7th":
                a = seventh_pool(a, anchor_mask)
                p = seventh_pool(p, pos_mask)
                
            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s)
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )
            # and now calculate the loss
            loss = loss_func(scores * scale, labels) 
            losses_train[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            train_loop.set_description(f"Epoch {epoch}")
            train_loop.set_postfix(loss=loss.item())
        
        # TEST LOADER -- no training model, only evaluating loss for the test loader
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        test_loop = tqdm(test_loader, leave=True)
        for i_batch, batch in enumerate(test_loop):
            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(device)     
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)
            pos_mask = batch[1][1].to(device)
            # extract token embeddings from model
            a = model(anchor_ids, attention_mask=anchor_mask)[0]  # all token embeddings
            p = model(pos_ids, attention_mask=pos_mask)[0]
            
            # get the pooled vectors
            if optimized_rep == "av":
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                
            elif optimized_rep == "cls":
                a = cls_pool(a, anchor_mask)
                p = cls_pool(p, pos_mask)
                
            elif optimized_rep == "sep":
                a = sep_pool(a, anchor_mask)
                p = sep_pool(p, pos_mask)
                
            elif optimized_rep == "7th":
                a = seventh_pool(a, anchor_mask)
                p = seventh_pool(p, pos_mask)
                
            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s)
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )
            # and now calculate the loss
            loss = loss_func(scores * scale, labels)  
            losses_test[epoch, i_batch] = loss.item()

            # update the TDQM progress bar
            test_loop.set_description(f"Epoch {epoch}")
            test_loop.set_postfix(loss=loss.item())


        ## evaluation 
        # TRAINING DATASET -- evaluation of abstract accuracy
        X_train = list(np.vstack(train_dataset.abs_sentences)[:, 0])
        y_train = list(np.vstack(train_dataset.abs_sentences)[:, 1])

        # knn accuracy
        embd_cls_train, embd_sep_train, embd_av_train = generate_embeddings(
            X_train, tokenizer, model, device, batch_size=256
        )
        knn_acc_train = knn_accuracy([embd_av_train, embd_cls_train, embd_sep_train], y_train)
        knn_accuracies_train[epoch,:]= knn_acc_train


        # TEST DATASET -- evaluation of abstract accuracy
        X_test = list(np.vstack(test_dataset.abs_sentences)[:, 0])
        y_test = list(np.vstack(test_dataset.abs_sentences)[:, 1])

        # knn accuracy
        embd_cls_test, embd_sep_test, embd_av_test = generate_embeddings(
            X_test, tokenizer, model, device, batch_size=256
        )
        knn_acc_test = knn_accuracy([embd_av_test, embd_cls_test, embd_sep_test], y_test)
        knn_accuracies_test[epoch,:]= knn_acc_test

        
    return losses_train, losses_test, knn_accuracies_train, knn_accuracies_test




def train_loop_with_projection_head(
    model,
    loader,
    device,
    titles_abstracts_together,
    tokenizer,
    label_mask,
    labels_acc,
    optimized_rep="av",
    n_epochs=1,
    lr=2e-5,
    eval_metric= "knn",
    return_seventh=False,
    return_embeddings= False,
):
    assert optimized_rep in [
        "av",
        "cls",
        "sep",
        "7th",
    ], "Not valid `optimized_rep`. Choose from ['av', 'cls', 'sep', '7th']."

    assert eval_metric in [
        "knn",
        "lin",
    ], "Not valid `eval_metric`. Choose from ['knn', 'lin']."

    model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value, it is the inverse of the temperature
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs 
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))
    accuracies = []
    for epoch in range(n_epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(device)     # this are all anchor abstracts from the batch,len(anchor_ids)= len(batch)
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)       # this each positive pair from each anchor, all in one array, also len(batch)
            pos_mask = batch[1][1].to(device)
            # forward pass
            z_a, a = model(anchor_ids, attention_mask=anchor_mask, optimized_rep=optimized_rep)  
            z_p, p = model(pos_ids, attention_mask=pos_mask, optimized_rep=optimized_rep)
            
            # # get the mean pooled vectors
            # if optimized_rep == "av":
            #     a = mean_pool(a, anchor_mask)
            #     p = mean_pool(p, pos_mask)
                
            # elif optimized_rep == "cls":
            #     a = cls_pool(a, anchor_mask)
            #     p = cls_pool(p, pos_mask)
                
            # elif optimized_rep == "sep":
            #     a = sep_pool(a, anchor_mask)
            #     p = sep_pool(p, pos_mask)
                
            # elif optimized_rep == "7th":
            #     a = seventh_pool(a, anchor_mask)
            #     p = seventh_pool(p, pos_mask)
                
            # calculate the cosine similarities 
            scores = torch.stack(
                [cos_sim(z_a_i.reshape(1, z_a_i.shape[0]), z_p) for z_a_i in z_a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )  # I think that the labels are just the label of which pair it is. 0 for the first pair, 1 for the second...
            # my guess is that they are used in the loss to know which of the cosine similarities should be high
            # and which low

            # and now calculate the loss
            loss = loss_func(scores * scale, labels) 
            losses[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            
        ## evaluation
        if return_seventh == True:
            (
                embedding_cls,
                embedding_sep,
                embedding_av,
                embedding_7th,
            ) = generate_embeddings(
                titles_abstracts_together,
                tokenizer,
                model.model,
                device,
                batch_size=256,
                return_seventh=True,
            )
            if eval_metric == "knn":
                acc = knn_accuracy(
                    [
                        embedding_av[label_mask],
                        embedding_cls[label_mask],
                        embedding_sep[label_mask],
                        embedding_7th[label_mask],
                    ],
                    labels_acc,
                )
            elif eval_metric == "lin":
                acc = logistic_accuracy([embedding_av[label_mask], embedding_cls[label_mask], embedding_sep[label_mask], embedding_7th[label_mask]], labels_acc)
    

        else:
            (
                embedding_cls,
                embedding_sep,
                embedding_av,
            ) = generate_embeddings(
                titles_abstracts_together,
                tokenizer,
                model.model,
                device,
                batch_size=256,
            )
            if eval_metric == "knn":
                acc = knn_accuracy(
                    [
                        embedding_av[label_mask],
                        embedding_cls[label_mask],
                        embedding_sep[label_mask],
                    ],
                    labels_acc,
                )
            elif eval_metric == "lin":
                acc = logistic_accuracy([embedding_av[label_mask], embedding_cls[label_mask], embedding_sep[label_mask]], labels_acc)


        accuracies.append(acc)
    if return_embeddings==True:
        if return_seventh == True:
            return losses, accuracies, embedding_cls, embedding_sep, embedding_av, embedding_7th
        if return_seventh == False:
            return losses, accuracies, embedding_cls, embedding_sep, embedding_av
    else:
        return losses, accuracies
    



def train_loop_train_test_split(
    model,
    loader,
    device,
    tokenizer,
    abstracts_eval_train,
    abstracts_eval_test,
    labels_eval_train,
    labels_eval_test,
    optimized_rep="av",
    n_epochs=1,
    lr=2e-5,
):
    assert optimized_rep in [
        "av",
        "cls",
        "sep",
        "7th",
    ], "Not valid `optimized_rep`. Choose from ['av', 'cls', 'sep', '7th']."

    model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value, it is the inverse of the temperature
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))
    accuracies = []
    for epoch in range(n_epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(
                device
            )  # this are all anchor abstracts from the batch,len(anchor_ids)= len(batch)
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(
                device
            )  # this each positive pair from each anchor, all in one array, also len(batch)
            pos_mask = batch[1][1].to(device)
            # extract token embeddings from BERT
            a = model(anchor_ids, attention_mask=anchor_mask)[
                0
            ]  # all token embeddings
            p = model(pos_ids, attention_mask=pos_mask)[0]

            # get the mean pooled vectors  -- put all of these ifs into a pool function (wraper) to which I pass, a, p the masks and the optimized rep
            if optimized_rep == "av":
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)

            elif optimized_rep == "cls":
                a = cls_pool(a, anchor_mask)
                p = cls_pool(p, pos_mask)

            elif optimized_rep == "sep":
                a = sep_pool(a, anchor_mask)
                p = sep_pool(p, pos_mask)

            elif optimized_rep == "7th":
                a = seventh_pool(a, anchor_mask)
                p = seventh_pool(p, pos_mask)

            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )  # I think that the labels are just the label of which pair it is. 0 for the first pair, 1 for the second...
            # my guess is that they are used in the loss to know which of the cosine similarities should be high
            # and which low

            # and now calculate the loss
            loss = loss_func(scores * scale, labels)
            losses[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        ## evaluation of the unseen test set for each epoch 
        (
            embedding_cls_train,
            embedding_sep_train,
            embedding_av_train,
        ) = generate_embeddings(
            abstracts_eval_train,
            tokenizer,
            model,
            device,
            batch_size=256,
        )

        (
            embedding_cls_test,
            embedding_sep_test,
            embedding_av_test,
        ) = generate_embeddings(
            abstracts_eval_test,
            tokenizer,
            model,
            device,
            batch_size=256,
        )

        knn = KNeighborsClassifier(
            n_neighbors=10, algorithm="brute", n_jobs=-1, metric="euclidean"
        )
        knn = knn.fit(embedding_av_train, labels_eval_train)
        acc = knn.score(embedding_av_test, labels_eval_test)

        accuracies.append(acc)

    return losses, accuracies
