import numpy as np
import pandas as pd
import random
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from collections import defaultdict
from pathlib import Path

from text_embeddings_src.eval_functions import KNNEval, MTEBEval

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


def poolerdecorator(name):
    """This function is a decorator. 
    When you use the decorator above another function, you can assign to that function an attribute calle `.sent_rep` with value `name`.
    
    """
    def decorator(fun):
        fun.sent_rep = name
        return fun
    return decorator


# pooling functions
@poolerdecorator("av")
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size())
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

@poolerdecorator("sep")
def sep_pool(token_embeds, attention_mask):
    ix = attention_mask.sum(1) - 1
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, ix, :]

@poolerdecorator("cls")
def cls_pool(token_embeds, attention_mask):
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, 0, :]


def train_loop(
    wrapped_model, 
    loader,  # training data loader
    device,
    eval_train_data,
    eval_train_labels,
    eval_test_data = None,  # None when eval is on MTEB or no train/test split
    eval_test_labels =None,
    eval_every_epochs =  True, #bool, {True, False} 
    eval_every_batches = 0, # int, 0 would be like none
    eval_function = KNNEval,
    pooler = mean_pool,
    eval_rep=None, # representation to evaluate, if None it is the same used by pooler
    dist_metric = "euclidean",
    mteb_saving_path = None,
    mteb_tasks = None, 
    n_epochs=1,
    lr=2e-5,
    scale = 20.0,  # we multiply similarity score by this scale value, it is the inverse of the temperature
):
    
    assert not eval_function == MTEBEval or (mteb_saving_path is not None and mteb_tasks is not None), "You forgot either the MTEB saving path or list of tasks for the MTEB evaluation."
    
    if eval_rep is None:
        eval_rep = pooler.sent_rep

    if eval_every_batches != 0: # if this happens, eval happens after every X batch and after the last batch
        eval_every_epochs = False # therefore, we set the evaluation after the full epoch to 0 to not evaluate twice after the last batch of the epoch


    ## training set up
    wrapped_model.model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
      
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(wrapped_model.model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs 
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))

    # initialize eval list
    if (eval_every_epochs != 0) | (eval_every_batches != 0):
        training_eval_results = defaultdict(list)

    ## training
    for epoch in range(n_epochs):
        wrapped_model.model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            ## train -- finished
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(device)     # this are all anchor abstracts from the batch,len(anchor_ids)= len(batch)
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(device)       # this each positive pair from each anchor, all in one array, also len(batch)
            pos_mask = batch[1][1].to(device)

            # get hidden state
            a = wrapped_model.get_outputs(input_ids=anchor_ids, attention_mask=anchor_mask)
            p = wrapped_model.get_outputs(input_ids = pos_ids, attention_mask=pos_mask)
            
            # get the mean pooled vectors  
            a = pooler(a, anchor_mask)
            p = pooler(p, pos_mask)

            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )  # the labels are just the "label" of which pair it is (0 for the first pair, 1 for the second)
            # they are used in the loss to know which of the cosine similarities should be high and which low

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
            if eval_every_batches != 0:
                if (i_batch % eval_every_batches == 0) | (
                    i_batch == len(loader)-1
                ):  
                    # add batch number to the results
                    training_eval_results["batch"].append(i_batch+len(loader)*epoch)

                    # path with batch number for saving MTEB results
                    mteb_saving_name = Path(f"results_epoch_{epoch}_batch_{i_batch}")

                    eval_results = eval_function( # some of these are needed for knn eval and some others for mteb
                        wrapped_model = wrapped_model,
                        device = device,
                        dataset = eval_train_data,
                        labels = eval_train_labels,
                        test_dataset = eval_test_data,
                        test_labels = eval_test_labels,
                        eval_rep= eval_rep, 
                        dist_metric = dist_metric,
                        tasks = mteb_tasks,
                        path_to_save= mteb_saving_path / mteb_saving_name,
                    )
                    [training_eval_results[k].append(v) for k, v in eval_results.items()]
                    wrapped_model.model.train()



        if eval_every_epochs != 0:
            print("eval_epoch", epoch)
            
            # add epoch number to the results
            training_eval_results["epoch"].append(epoch)

            # same code as above for the batches
            mteb_saving_name = Path(f"results_epoch_{epoch}")

            eval_results = eval_function( # some of these are needed for knn eval and some others for mteb
                    wrapped_model = wrapped_model,
                    device = device,
                    dataset = eval_train_data,
                    labels = eval_train_labels,
                    test_dataset = eval_test_data,
                    test_labels = eval_test_labels,
                    eval_rep= eval_rep, 
                    dist_metric = dist_metric,
                    tasks = mteb_tasks,
                    path_to_save= mteb_saving_path / mteb_saving_name,
            )
            [training_eval_results[k].append(v) for k, v in eval_results.items()]
            wrapped_model.model.train()



    if (eval_every_epochs != 0) | (eval_every_batches != 0):
        # convert results to dataframe
        df_training_eval_results = pd.DataFrame(training_eval_results)

        # transform every knn/lin value from an array to a single number, if only one rep was evaluated (array([0.6]) --> 0.6)
        if "knn" in training_eval_results.keys():
            df_training_eval_results["knn"] = df_training_eval_results["knn"].apply(lambda x: x[0] if len(x) == 1 else x)
        if "lin" in training_eval_results.keys():
            df_training_eval_results["lin"] = df_training_eval_results["lin"].apply(lambda x: x[0] if len(x) == 1 else x)

        return losses, df_training_eval_results
    else:
        return losses