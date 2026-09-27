import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            verify_method="default",
            profile=False,

    ):
        # Acceptance length follows the benchmark/paper convention: the
        # number of output tokens produced by one draft-verification round.
        # A round that accepts ``accept_length`` draft tokens also emits the
        # already target-sampled root/bonus token, so its acceptance length is
        # ``accept_length + 1``.  Keep the draft-only count separately to make
        # the two commonly used metrics unambiguous.
        verification_rounds = 0
        total_accept_length = 0
        depth_accept={k:0 for k in range(6)}

        total_accepted_draft_tokens = 0
        acceptance_lengths = []
        accepted_draft_lengths = []
        traversal_stats = {
            "visited_nodes": 0,
            "rejected_nodes": 0,
            "refresh_syncs": 0,
            "refreshed_edges": 0,
            "compiled_residuals": 0,
        }
        self.last_eagenerate_metrics = {
            "verify_method": verify_method.lower(),
            "verification_rounds": 0,
            "total_accept_length": 0,
            "total_generated_tokens": 0,
            "total_accepted_draft_tokens": 0,
            "average_accept_length": 0.0,
            "average_accepted_draft_tokens": 0.0,
            "acceptance_lengths": [],
            "accepted_draft_lengths": [],
        }
        profile_events = {} if profile and torch.cuda.is_available() else None
        self.last_eagenerate_profile_events = profile_events

        def start_profile_phase():
            if profile_events is None:
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def end_profile_phase(name, start_event):
            if start_event is None:
                return
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            profile_events.setdefault(name, []).append(
                (start_event, end_event)
            )

        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        phase_start = start_profile_phase()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, tree_info = initialize_tree(
            input_ids,
            self,
            past_key_values,
            logits_processor,
            verify_method,
            return_tree_info=True,
        )
        end_profile_phase("initial_tree", phase_start)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        idx = -1
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask
            method = verify_method.lower()

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            phase_start = start_profile_phase()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                compact_logits=method in {
                    "univer", "greedy", "rrsw", "traversal", "traversal_verification"
                },
            )
            end_profile_phase("target_decode", phase_start)
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            phase_start = start_profile_phase()

            if method == "default":
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
            elif method == "univer":
                best_candidate, accept_length, sample_p = evaluate_posterior2(
                    logits,
                    candidates,
                    retrieve_indices,
                    logits_processor,
                    tree_info,
                )
            elif method == "rrsw":
                best_candidate, accept_length, sample_p = evaluate_posterior3(
                    logits, candidates, logits_processor, tree_info, retrieve_indices
                )
            elif method in {"traversal", "traversal_verification"}:
                best_candidate, accept_length, sample_p = evaluate_posterior4(
                    logits, candidates, logits_processor, tree_info, retrieve_indices
                )
                round_traversal_stats = tree_info.get("_traversal_stats", {})
                for name, value in round_traversal_stats.items():
                    traversal_stats[name] = traversal_stats.get(name, 0) + value

            elif method == "greedy":
                best_candidate, accept_length, sample_p = evaluate_posterior5(
                    logits, candidates, logits_processor, tree_info, retrieve_indices
                )

            else:
                raise ValueError(f"Unsupported verify_method: {verify_method!r}")
            end_profile_phase("verification", phase_start)

            accepted_draft_count = int(accept_length)
            round_accept_length = accepted_draft_count + 1
            verification_rounds += 1
            total_accepted_draft_tokens += accepted_draft_count
            total_accept_length += round_accept_length
            # for i in range(min(round_accept_length,5)):
            #     depth_accept[i]+=1
            depth_accept[round_accept_length]=depth_accept.get(round_accept_length,0)+1
            accepted_draft_lengths.append(accepted_draft_count)
            acceptance_lengths.append(round_accept_length)
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            round_input_length = input_ids.shape[1]
            phase_start = start_profile_phase()
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token, tree_info = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
                verify_method,
                return_tree_info=True,
            )
            end_profile_phase("update_and_draft", phase_start)

            # Check only tokens emitted by this round.  The previous code
            # copied the complete, growing output to the CPU twice per round.
            emitted_ids = input_ids[0, round_input_length:]
            stop_mask = emitted_ids == self.tokenizer.eos_token_id
            if is_llama3:
                stop_mask.logical_or_(emitted_ids == stop_token_id)
            if bool(stop_mask.any()):
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        average_accept_length = (
            total_accept_length / verification_rounds if verification_rounds else 0.0
        )
        average_accepted_draft_tokens = (
            total_accepted_draft_tokens / verification_rounds
            if verification_rounds else 0.0
        )
        self.last_eagenerate_metrics = {
            "verify_method": verify_method.lower(),
            "verification_rounds": verification_rounds,
            "total_accept_length": total_accept_length,
            "total_generated_tokens": new_token,
            "total_accepted_draft_tokens": total_accepted_draft_tokens,
            "average_accept_length": average_accept_length,
            "average_accepted_draft_tokens": average_accepted_draft_tokens,
            "acceptance_lengths": acceptance_lengths,
            "accepted_draft_lengths": accepted_draft_lengths,
            "traversal_stats": traversal_stats,
            "depth_accept":depth_accept,
        }
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # Use the same target-model forward path as ``eagenerate``.  Calling
        # ``self.base_model`` here goes through a different CausalLM wrapper
        # (including an unconditional float32 logits cast), which makes the
        # in-process baseline less comparable to speculative decoding.
        _, target_logits, _ = self(
            input_ids,
            past_key_values=past_key_values,
            output_orig=True,
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = target_logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = target_logits[:, -1:].argmax(dim=-1)
            # With ``device_map='auto'`` the input embedding and LM head can
            # live on different GPUs.  Sampling follows the logits onto the
            # LM-head device, but token ids must return to the input/embedding
            # device before the next forward pass and sequence concatenation.
            input_id = input_id.to(input_ids.device)
            _, target_logits, _ = self(
                input_id,
                past_key_values=past_key_values,
                output_orig=True,
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            emitted_id = input_ids[0, -1]
            should_stop = emitted_id == self.tokenizer.eos_token_id
            if is_llama3:
                should_stop = should_stop | (emitted_id == stop_token_id)
            if bool(should_stop):
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            verify_method="default",

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, tree_info = initialize_tree(
            input_ids,
            self,
            past_key_values,
            logits_processor,
            verify_method,
            return_tree_info=True,
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask
            method = verify_method.lower()

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                compact_logits=method in {
                    "univer", "greedy", "rrsw", "traversal", "traversal_verification"
                },
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            if method == "univer":
                best_candidate, accept_length, sample_p = evaluate_posterior2(
                    logits,
                    candidates,
                    retrieve_indices,
                    logits_processor,
                    tree_info,
                )
            elif method == "default":
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
            elif method == "rrsw":
                best_candidate, accept_length, sample_p = evaluate_posterior3(
                    logits, candidates, logits_processor, tree_info, retrieve_indices
                )
            elif method in {"traversal", "traversal_verification"}:
                best_candidate, accept_length, sample_p = evaluate_posterior4(
                    logits, candidates, logits_processor, tree_info, retrieve_indices
                )
            elif method == "greedy":
                best_candidate, accept_length, sample_p = evaluate_posterior5(
                    logits, candidates, logits_processor, tree_info, retrieve_indices
                )
            else:
                raise ValueError(f"Unsupported verify_method: {verify_method!r}")

            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token, tree_info = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
                verify_method,
                return_tree_info=True,
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
