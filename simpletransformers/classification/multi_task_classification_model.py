import logging
import numpy as np
import random
import warnings
from multiprocessing import cpu_count

import torch
from simpletransformers.classification import ClassificationModel
from simpletransformers.custom_models.models import BertForMultiTaskSequenceClassification
from transformers import (
    BertConfig,
    BertTokenizer,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class MultiTaskClassificationModel(ClassificationModel):
    def __init__(
        self,
        model_name,
        num_labels_1=None,
        num_labels_2=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a MultiTaskClassification model.

        Args:
            model_name: Default Transformer model name or path to a directory 
                         containing Transformer model file (pytorch_nodel.bin).
            num_labels_1 (optional): The number of labels or classes
                                      for the first classification task
            num_additional_labels: The number of labels or classes 
                                      for the second classification task.
            args (optional): Default args will be used if this parameter is not provided. 
                          If provided, it should be a dict containing the args 
                          that should be changed in the default args.
            use_cuda (optional): Use GPU if available. 
                          Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. 
                                  Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, 
                           resume_download, cache_dir and other options 
                           specific to the 'from_pretrained' implementation 
                           where this will be supplied.
        """

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args


        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        # For hyperparameter tuning
        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False


        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)



        if self.args.labels_list_1:
            if num_labels_1:
                assert num_labels_1 == len(self.args.labels_list_1)
            if self.args.labels_map_1:
                try:
                    assert list(self.args.labels_map_1.keys()) == self.args.labels_list_1
                except AssertionError:
                    assert [int(key) for key in list(self.args.labels_map_1.keys())] == self.args.labels_list_1
                    self.args.labels_map_1 = {int(key): value for key, value in self.args.labels_map_1.items()}
            else:
                self.args.labels_map_1 = {label: i for i, label in enumerate(self.args.labels_list_1)}
        else:
            len_labels_list_1 = 2 if not num_labels_1 else num_labels_1
            self.args.labels_list_1 = [i for i in range(len_labels_list_1)]



        if self.args.labels_list_2:
            if num_labels_2:
                assert num_labels_2 == len(self.args.labels_list_2)
            if self.args.labels_map_2:
                try:
                    assert list(self.args.labels_map_2.keys()) == self.args.labels_list_2
                except AssertionError:
                    assert [int(key) for key in list(self.args.labels_map_2.keys())] == self.args.labels_list_2
                    self.args.labels_map_2 = {int(key): value for key, value in self.args.labels_map_2.items()}
            else:
                self.args.labels_map_2 = {label: i for i, label in enumerate(self.args.labels_list_2)}
        else:
            len_labels_list_2 = 2 if not num_labels_2 else num_labels_2
            self.args.labels_list_2 = [i for i in range(len_labels_list_2)]


        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        # only if we provide the num_labels and num_additional_labels as function arguments,
        # they will be assigned;
        # otherwise, if we provide config of the already trained model, the num_labels and
        # num_additional_labels will be taken from there
        # if we do not provide anything, the num_labels will default to 2 (due to the transformers library),
        # and num_additional_labels will not be set (should maybe fix this later?)
        if num_labels and num_additional_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=num_labels, **self.args["config"])
            self.num_labels = num_labels # we don't seem to use this later on?
            self.config.num_additional_labels = num_additional_labels # have to manually set this attribute in config
                                                                      # because I don't want to modify the transformers
                                                                      # library too
        else:
            self.config = config_class.from_pretrained(model_name, **self.args["config"])
            self.num_labels = self.config.num_labels # 2 by default (unless we provide config in which
                                                             # num_labels is specified)
                                                            
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)

        self.results = {}
        self.additional_results = {} # for the second classification task

        if not use_cuda:
            self.args["fp16"] = False


        self.tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=self.args["do_lower_case"], **kwargs
        )

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None

    def train_model(
        self,
        train_df,
        multi_task=True,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        return super().train_model(
            train_df,
            multi_task=multi_task,
            multi_label=multi_label,
            output_dir=output_dir,
            show_running_loss=show_running_loss,
            args=args,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

    def eval_model(self, eval_df, multi_label=False, multi_task=True, output_dir=None, verbose=False, silent=False, **kwargs):
        return super().eval_model(
            eval_df, multi_label=multi_label, multi_task=multi_task, output_dir=output_dir, verbose=verbose, silent=silent, **kwargs
        )

    def load_and_cache_examples(
        self, examples, evaluate=False, no_cache=False, multi_label=False, multi_task=True, verbose=True, silent=False
    ):
        return super().load_and_cache_examples(
            examples, evaluate=evaluate, no_cache=no_cache, multi_label=multi_label, multi_task=multi_task, verbose=verbose, silent=silent
        )

    def predict(self, to_predict, multi_label=False, multi_task=True):
        return super().predict(to_predict, multi_label=multi_label, multi_task=multi_task)
