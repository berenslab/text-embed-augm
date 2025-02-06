from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
import datasets
import numpy as np
from transformers import AutoModel, AutoTokenizer
from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer, models

from text_embeddings_src.embeddings import generate_embeddings, generate_embeddings_embed_layer

# TODO: docstrings and comments

class ModelWrapper(ABC):
    """
    I should always define a self.model
    """
    # Q?: Can I not do this to "enforce" a type of init?
    def __init__(self, model, tokenizer):
        self.checkpoint = model.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer  


    @abstractmethod
    def encode_dataset(self, data, device):
        """For kNN and linear evaluation"""
        # return embedding_cls, embedding_sep, embedding_av
        pass

    @abstractmethod
    def ST_wrapper(self):
        """For MTEB evaluation"""
        # to transform it to a SentenceTransformer for MTEB eval
        # return ST_model
        pass

    @abstractmethod
    def get_outputs(self, input_ids, attention_mask):
        """For the train_loop.
        To get the latent representation as a single output. For using in train_loop.
        output : all token embeddings
        """
        # output = self.model(input_ids, attention_mask=attention_mask)[0]
        ...
        # return output
        pass
    


class HFModelWrapper(ModelWrapper):
    """ONLY FOR PRE-TRAINED MODELS"""
    def __init__(self, model, tokenizer):
        """"""
        self.checkpoint = model.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer      

    def encode_dataset(self, data, device):
        """For knn and linear evaluation"""
        # TODO: potentially make it return only the representation that is being optimized (?)
        (
            embedding_cls,
            embedding_sep,
            embedding_av,
        ) = generate_embeddings(data, 
                        self.tokenizer, 
                        self.model, 
                        device, 
                        batch_size=256, 
                        return_seventh = False)
        return embedding_cls, embedding_sep, embedding_av

    def get_outputs(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)[0]
        return output
    
    def ST_wrapper(self):
        ST_model = SentenceTransformer(self.checkpoint)
        return ST_model


class FineTunedHFModelWrapper(HFModelWrapper):
    """FOR HF MODELS THAT ARE ALREADY/GOING TO BE FINE-TUNED"""
    def __init__(self, model, tokenizer, checkpoint= None):
        if checkpoint is None:
            checkpoint = "bert-base-uncased"
        self.base_checkpoint = checkpoint
        self.model = model
        self.tokenizer = tokenizer   

    def ST_wrapper(self):
        # Create a new SentenceTransformer model
        new_modules = []

        # Wrap your custom base model in a Transformer module
        transformer_model = models.Transformer(
            model_name_or_path=self.base_checkpoint,  # None gives an error,
            # so I initialize with the pre-trained model that
            # will be substituted by my fine-tuned model below
            max_seq_length=384,  # You can adjust this as needed
            do_lower_case=False,  # Adjust based on your tokenizer
        )
        # Replace the auto_model in the Transformer wrapper with your custom model
        transformer_model.auto_model = self.model
        # Set the tokenizer
        transformer_model.tokenizer = self.tokenizer

        # Add the wrapped model as the first module
        new_modules.append(transformer_model)

        # Add Pooling layer
        pooling_model = models.Pooling(
            word_embedding_dimension=self.model.config.hidden_size,
            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_sqrt_len_tokens=False,
        )
        new_modules.append(pooling_model)

        # Add Normalize layer
        new_modules.append(models.Normalize())

        # Create the new SentenceTransformer model
        ST_model = SentenceTransformer(modules=new_modules)
        return ST_model


class ModelWithProjectionHead(nn.Module):
    def __init__(self, checkpoint, pooler, in_dim=768, feat_dim=128, hidden_dim=512):
        super().__init__()  # to inherit from the nn.Module
        self.pooler = pooler  # pooler function
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # load model
        if checkpoint == "allenai/specter2_base":
            self.backbone = AutoAdapterModel.from_pretrained(checkpoint)
            # add adapter proximity
            self.backbone.load_adapter(
                "allenai/specter2",
                source="hf",
                load_as="specter2",
                set_active=True,
            )
        else:
            self.backbone = AutoModel.from_pretrained(
                checkpoint
            )  

        # add projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feat_dim),
        )

    def forward(self, input_ids, attention_mask):
        """
        pooler : {mean_pool, cls_pool, sep_pool, seventh_pool?}
        """
        # Extract outputs from the body
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

        # pooling
        h = self.pooler(outputs, attention_mask)

        # Add custom layers
        z = self.projection_head(h)  # .view(-1,768)

        return z


class ModelWithProjectionHeadWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        """
        model : full model with the projection head
        self.model : full model with the projection head
        self.backbone : model without the projection head
        """
        self.checkpoint = model.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer 
        self.backbone = model.backbone

    def get_outputs(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs
    
    def encode_dataset(self, data, device):
        """For knn and linear evaluation.
        It gives back the representation after the backbone, i.e., before the projection head.
        """
        (
            embedding_cls,
            embedding_sep,
            embedding_av,
        ) = generate_embeddings(data, 
                        self.tokenizer, 
                        self.backbone, 
                        device, 
                        batch_size=256, 
                        return_seventh = False)
        return embedding_cls, embedding_sep, embedding_av
    
    def ST_wrapper(self):
        # this function was not needed so far therefore not implemented
        pass




class EmbeddingOnlyModel(torch.nn.Module):
    """Create a new model with only the embedding layer.
    Valid function for both the layer and the module (04/09/2024)
    
    Parameters
    ----------
    """
    def __init__(self, model_name_or_embeddings):
        super().__init__()
        if isinstance(model_name_or_embeddings, str):
            # If a string is provided, load the pretrained model and extract embeddings
            pretrained_model = AutoModel.from_pretrained(
                model_name_or_embeddings
            )
            self.embeddings = pretrained_model.embeddings.word_embeddings
        else:
            # If embeddings are provided directly, use them
            self.embeddings = model_name_or_embeddings

    def forward(self, input_ids):
        return self.embeddings(input_ids)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))

    @classmethod
    def from_pretrained(cls, load_directory, base_model="bert-base-uncased"):
        # Q?: is it a problem that is bert-base-uncased when using a pre-trained embedding layer from MPNet?
        # Load the state dict
        state_dict = torch.load(os.path.join(load_directory, "model.pt"))

        # Create a new instance of the model with a dummy model name
        # We'll replace the embeddings with the loaded state dict
        model = cls(base_model)

        # Load the state dict
        model.load_state_dict(state_dict)

        return model



# Q?: is it possible to make this the ST_wrapper function of class above? I think not.
class MyEmbeddingSentenceModel: 
    """Sentence embedding model using only the embedding layer of a transformer.
    Uses class EmbeddingOnlyModel (see above) and puts it in the format of Sentence Transformers, to be able to evaluate it in the MTEB tasks.
    """
    def __init__(self, model, tokenizer, pooler):
        self.tokenizer = tokenizer
        self.model = model
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.pooler = pooler

        self.model.to(self.device)

    @torch.no_grad()  # Q?: what is the difference between no_grad() and inference_mode()?
    def encode(self, input_texts, batch_size=None, **kwargs):
        inputs = self.tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        dataset = datasets.Dataset.from_dict(inputs)
        dataset.set_format(type="torch", output_all_columns=True)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=10
        )
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()} # it used to be device only (not self.device) but I think it only wroked bc of jupyter having device defined somewhere else
                outputs = self.model(batch["input_ids"])
                embdd = self.pooler(outputs, batch["attention_mask"])
                embeddings.append(embdd.detach().cpu().numpy())

        embeddings = np.vstack(embeddings)
        return embeddings 



class EmbeddingOnlyModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, pooler):
        "model : instance of EmbeddingOnlyModel"
        self.model = model
        self.tokenizer = tokenizer
        self.pooler = pooler


    def get_outputs(self, input_ids, **kwargs):
        """
        **kwargs : attention_mask is also always passed to get_outputs but in this case it is not used (because it is only the embedding layer)
        """
        outputs = self.model(input_ids)
        return outputs
    
    def encode_dataset(self, data, device):
        """For knn and linear evaluation"""
        (
            embedding_cls,
            embedding_sep,
            embedding_av,
        ) = generate_embeddings_embed_layer(data, 
                        self.tokenizer, 
                        self.model, 
                        device, 
                        batch_size=256, 
                        return_seventh = False)
        return embedding_cls, embedding_sep, embedding_av

    
    def ST_wrapper(self):
        ST_model = MyEmbeddingSentenceModel(model = self.model,
                                            tokenizer= self.tokenizer,
                                            pooler = self.pooler)
        return ST_model




# extra function
def check_models_equal(original_model, loaded_model):
    """Checks if models have identical parameters. 
    The == operator does not work because two separately instantiated model objects will always be considered different, even if they have identical parameters.
    
    """
    # Check if state dictionaries are equal
    original_state_dict = original_model.state_dict()
    loaded_state_dict = loaded_model.state_dict()

    # This checks if all keys and tensor values are the same
    are_equal = all(
        torch.equal(original_state_dict[key], loaded_state_dict[key])
        for key in original_state_dict
    )

    print(f"Models have identical parameters: {are_equal}")

    # # You can also check individual layers if needed
    # print(torch.equal(original_model.embeddings.word_embeddings.weight,
    #                   loaded_model.embeddings.word_embeddings.weight))