#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_phi.modeling_phi import PhiModel, PhiForCausalLM, CausalLMHead, CausalLMLoss
from .modeling_phi.configuration_phi import PhiConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ChatUniVi.model.arch import MetaModel, ChatUniViMetaForCausalLM


class ChatUniViConfig(PhiConfig):
    model_type = "ChatUniViPhi2"


class ChatUniViPhiModel(MetaModel, PhiModel):
    config_class = ChatUniViConfig

    def __init__(self, config: PhiConfig):
        super(ChatUniViPhiModel, self).__init__(config)


class ChatUniViPhiForCausalLM(PhiForCausalLM, ChatUniViMetaForCausalLM):
    config_class = ChatUniViConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.config = config
        self.transformer = ChatUniViPhiModel(config)
        self.lm_head = CausalLMHead(config)
        self.loss = CausalLMLoss()

        self.post_init()

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            try:
                loss = loss_fct(shift_logits, shift_labels)
            except:
                loss = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            
        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("ChatUniViPhi2", ChatUniViConfig)
AutoModelForCausalLM.register(ChatUniViConfig, ChatUniViPhiForCausalLM)
