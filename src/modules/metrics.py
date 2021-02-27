# Copyright (c) Facebook, Inc. and its affiliates.
"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy, vqa_accuracy,
r@1 etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``@registry.register_metric('name')``.

Example::

    import torch

    from mmf.common.registry import registry
    from mmf.modules.metrics import BaseMetric

    @registry.register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_config:
        pythia:
            metrics:
            - type: some
              params:
                some_param: a
"""

import collections
import warnings

import torch
from src.common.registry import registry


class Metrics:
    """Internally used by MMF, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (ListConfig): List of DictConfigs where each DictConfig
                                        specifies name and parameters of the
                                        metrics used.
    """

    def __init__(self, metric_list):
        if not isinstance(metric_list, collections.abc.Sequence):
            metric_list = [metric_list]

        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        self.required_params = {"dataset_name", "dataset_type"}
        for metric in metric_list:
            params = {}
            if isinstance(metric, collections.abc.Mapping):
                if "type" not in metric:
                    raise ValueError(
                        f"Metric {metric} needs to have 'type' attribute "
                        + "or should be a string"
                    )
                metric_type = key = metric.type
                params = getattr(metric, "params", {})
                # Support cases where uses need to give custom metric name
                if "key" in metric:
                    key = metric.key

                # One key should only be used once
                if key in metrics:
                    raise RuntimeError(
                        f"Metric with type/key '{metric_type}' has been defined more "
                        + "than once in metric list."
                    )
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )
                metric_type = key = metric

            metric_cls = registry.get_metric_class(metric_type)
            if metric_cls is None:
                raise ValueError(
                    f"No metric named {metric_type} registered to registry"
                )

            metric_instance = metric_cls(**params)
            metric_instance.name = key

            metrics[key] = metric_instance
            self.required_params.update(metrics[key].required_params)

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}

        dataset_type = sample_list.dataset_type
        dataset_name = sample_list.dataset_name

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                key = f"{dataset_type}/{dataset_name}/{metric_name}"
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

                if not isinstance(values[key], torch.Tensor):
                    values[key] = torch.tensor(values[key], dtype=torch.float)
                else:
                    values[key] = values[key].float()

                if values[key].dim() == 0:
                    values[key] = values[key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values



class BaseMetric:
    """Base class to be inherited by all metrics registered to MMF. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.required_params = ["scores", "targets"]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``SampleList`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample_list (SampleList): SampleList provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value


@registry.register_metric("textvqa_accuracy")
class TextVQAAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("textvqa_accuracy")
        import src.utils.m4c_evaluators as evaluators

        self.evaluator = evaluators.TextVQAAccuracyEvaluator()
        self.required_params = ["scores", "answers", "context_tokens"]
        self.gt_key = "answers"

    def calculate(self, sample_list, model_output, *args, **kwargs):
        answer_processor = registry.get(sample_list.dataset_name + "_answer_processor")

        batch_size = sample_list.context_tokens.size(0)
        pred_answers = model_output["scores"].argmax(dim=-1)
        context_tokens = sample_list.context_tokens.cpu().numpy()
        answers = sample_list.get(self.gt_key).cpu().numpy()
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        from src.utils.distributed import byte_tensor_to_object
        from src.utils.text import word_tokenize

        for idx in range(batch_size):
            tokens = byte_tensor_to_object(context_tokens[idx])
            answer_words = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )

            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            gt_answers = byte_tensor_to_object(answers[idx])
            predictions.append({"pred_answer": pred_answer, "gt_answers": gt_answers})

        accuracy = self.evaluator.eval_pred_list(predictions)
        accuracy = torch.tensor(accuracy).to(sample_list.context_tokens.device)

        return accuracy
    