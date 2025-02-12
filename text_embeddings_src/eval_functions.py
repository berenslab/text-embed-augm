import numpy as np
import mteb
from text_embeddings_src.metrics import knn_accuracy, linear_accuracy




# TODO: make outputs of all functions be a dictionary like Nik said
# TODO: docstrings and comments



def KNNEval(*, wrapped_model, device, dataset, labels, test_dataset = None, test_labels = None, eval_rep, dist_metric = "euclidean",**kwargs):
    # NO TRAIN/TEST SPLIT IN TRAINING
    if test_dataset is None:  
        # ENH: this code is inneficient since I have twice the same embeddings in memory.
        # encode dataset
        embedding_cls, embedding_sep, embedding_av = wrapped_model.encode_dataset(data = dataset, device = device)
            
        embedding_rep_dict = {"cls" : (embedding_cls,),
                        "sep" : (embedding_sep,),
                        "av" : (embedding_av,),}
        
        # eval code 
        eval_results = knn_accuracy(embedding_rep_dict[eval_rep][0], 
                    labels, 
                    test_embeddings = None,
                    test_labels = None, 
                    test_size=0.1, 
                    k = 10, 
                    rs=42, 
                    metric=dist_metric)
        
    # TRAIN/TEST SPLIT IN TRAINING    
    else: 
        # ENH: this code is inneficient since I have twice the same embeddings in memory.
        # encode train set
        embedding_cls_train, embedding_sep_train, embedding_av_train = wrapped_model.encode_dataset(data = dataset, device = device)
        # encode test set 
        embedding_cls_test, embedding_sep_test, embedding_av_test = wrapped_model.encode_dataset(data = test_dataset, device = device)

        embedding_rep_dict = {"cls" : (embedding_cls_train,embedding_cls_test),
                                "sep" : (embedding_sep_train,embedding_sep_test),
                                "av" : (embedding_av_train, embedding_av_test)}

        # eval code 
        eval_results = knn_accuracy(embedding_rep_dict[eval_rep][0], 
                    labels, 
                    test_embeddings = embedding_rep_dict[eval_rep][1], 
                    test_labels = test_labels, 
                    test_size=0.1,
                    k = 10, 
                    rs=42, 
                    metric=dist_metric)
            

    return {"knn":eval_results}
    


def LinEval(*, wrapped_model, device, dataset, labels, test_dataset = None, test_labels = None, eval_rep="av", **kwargs):
    # NO TRAIN/TEST SPLIT IN TRAINING
    if test_dataset is None:  
        # ENH: this code is inneficient since I have twice the same embeddings in memory.
        # encode dataset
        embedding_cls, embedding_sep, embedding_av = wrapped_model.encode_dataset(data = dataset, device = device) 
            
        embedding_rep_dict = {"cls" : (embedding_cls),
                        "sep" : (embedding_sep),
                        "av" : (embedding_av),}
        
        # eval code 
        eval_results = linear_accuracy(embedding_rep_dict[eval_rep][0], 
                    labels, 
                    test_embeddings = None,
                    test_labels = None, 
                    test_size=0.1, 
                    rs=42)
        
    # TRAIN/TEST SPLIT IN TRAINING    
    else: 
        # ENH: this code is inneficient since I have twice the same embeddings in memory.
        # encode train set
        embedding_cls_train, embedding_sep_train, embedding_av_train = wrapped_model.encode_dataset(data = dataset, device = device)
        # encode test set 
        embedding_cls_test, embedding_sep_test, embedding_av_test = wrapped_model.encode_dataset(data = test_dataset, device = device)

        embedding_rep_dict = {"cls" : (embedding_cls_train,embedding_cls_test),
                                "sep" : (embedding_sep_train,embedding_sep_test),
                                "av" : (embedding_av_train, embedding_av_test)}

        # eval code 
        eval_results = linear_accuracy(embedding_rep_dict[eval_rep][0], 
                    labels, 
                    test_embeddings = embedding_rep_dict[eval_rep][1], 
                    test_labels = test_labels, 
                    test_size=0.1,
                    rs=42)
            

    return {"lin":eval_results}
    


def MTEBEval(*, wrapped_model, tasks, path_to_save, **kwargs):
    """
    wrapped_model: wrapped_wrapped_model
    """
    # set up tasks and eval
    tasks = mteb.get_tasks(
        tasks=tasks
    )
    evaluation = mteb.MTEB(tasks=tasks)

    # wrap model
    ST_wrapped_model = wrapped_model.ST_wrapper()

    # build evaluation
    mteb_results = evaluation.run(
            ST_wrapped_model,
            output_folder= path_to_save, # during eval, results are also saved
            overwrite_results= True, # default False, if False it skips the evaluation if results folder exists!
            ) 
    
    
    # unwrap the dict into the dictionary in the way I want it
    dict_results = dict()
    for task in mteb_results:
        dict_results[task.task_name] = task.scores["test"][0]["main_score"]


    return dict_results             

