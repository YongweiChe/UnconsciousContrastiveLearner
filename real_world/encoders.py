"""Uniform wrappers around the pretrained contrastive models.

Each encoder exposes
    embed(modality, items) -> [n, d] tensor  (normalized later by real_world.embed)
    scales()               -> {"<mod>|<mod>": logit scale of that trained pair}

Modalities are "image", "audio" and "text". The logit scale is the temperature the
pair was trained with, i.e. the critic is f(x, y) = scale * x^T y on unit vectors.
That is the critic Assumption 2 / Lemma 1 refer to, and the one the Monte Carlo
estimator uses.

Libraries are imported lazily so each encoder only needs its own dependencies.
"""

import torch


def pair_key(m1, m2):
    return "|".join(sorted((m1, m2)))


class CLIP:
    modalities = ("image", "text")

    def __init__(self, device, model="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    def embed(self, modality, items):
        if modality == "image":
            from PIL import Image

            x = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in items]).to(self.device)
            return self.model.encode_image(x)
        return self.model.encode_text(self.tokenizer(list(items)).to(self.device))

    def scales(self):
        return {pair_key("image", "text"): self.model.logit_scale.exp().item()}


class CLAP:
    modalities = ("audio", "text")

    def __init__(self, device, fusion=False, ckpt=None):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=fusion, device=device)
        # With no path, laion_clap downloads its default AudioSet checkpoint for this fusion setting.
        self.model.load_ckpt(ckpt) if ckpt else self.model.load_ckpt()

    def embed(self, modality, items):
        items = list(items)
        if modality == "audio":
            return self.model.get_audio_embedding_from_filelist(x=items, use_tensor=True)
        # get_text_embedding mishandles a batch of one, so pad and drop.
        out = self.model.get_text_embedding(items if len(items) > 1 else items * 2, use_tensor=True)
        return out[: len(items)]

    def scales(self):
        return {pair_key("audio", "text"): self.model.model.logit_scale_a.exp().item()}


class ImageBind:
    modalities = ("image", "audio", "text")

    def __init__(self, device):
        from imagebind.models import imagebind_model

        self.device = device
        self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)

    def embed(self, modality, items):
        from imagebind import data
        from imagebind.models.imagebind_model import ModalityType

        items = list(items)
        key, load = {
            "image": (ModalityType.VISION, data.load_and_transform_vision_data),
            "audio": (ModalityType.AUDIO, data.load_and_transform_audio_data),
            "text": (ModalityType.TEXT, data.load_and_transform_text),
        }[modality]
        # The model's postprocessors normalize and then multiply by the logit scale;
        # real_world.embed re-normalizes, and the scales are recorded in scales().
        return self.model({key: load(items, self.device)})[key]

    def scales(self):
        from imagebind.models.imagebind_model import ModalityType

        def scale(modality):
            scaling = self.model.modality_postprocessors[modality][1]
            return scaling.log_logit_scale.exp().clamp(max=scaling.max_logit_scale).item()

        # ImageBind trains every modality against vision; the scale lives on the non-vision side.
        return {pair_key("image", "text"): scale(ModalityType.TEXT), pair_key("image", "audio"): scale(ModalityType.AUDIO)}


class LanguageBind:
    modalities = ("image", "audio", "text")

    def __init__(self, device, cache_dir="./cache_dir"):
        from languagebind import LanguageBind as LB
        from languagebind import LanguageBindImageTokenizer, transform_dict

        self.device = device
        clip_type = {"image": "LanguageBind_Image", "audio": "LanguageBind_Audio_FT"}
        self.model = LB(clip_type=clip_type, use_temp=False, cache_dir=cache_dir).to(device).eval()
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            "lb203/LanguageBind_Image", cache_dir=f"{cache_dir}/tokenizer_cache_dir"
        )
        self.transform = {m: transform_dict[m](self.model.modality_config[m]) for m in clip_type}

    def embed(self, modality, items):
        from languagebind import to_device

        items = list(items)
        if modality == "text":
            tokens = self.tokenizer(items, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
            return self.model({"language": to_device(tokens, self.device)})["language"]
        return self.model({modality: to_device(self.transform[modality](items), self.device)})[modality]

    def scales(self):
        # Each LanguageBind modality is trained against language with its own logit scale.
        return {pair_key(m, "text"): self.model.modality_scale[m].exp().item() for m in ("image", "audio")}


ENCODERS = {"clip": CLIP, "clap": CLAP, "imagebind": ImageBind, "languagebind": LanguageBind}
