import logging
import tarfile
from importlib import import_module, resources
from pathlib import Path

from PolyRL.models.gpt2 import (
    create_gpt2_actor,
    create_gpt2_actor_critic,
    create_gpt2_critic,
)

from PolyRL.models.gru import (
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
)
from PolyRL.models.llama2 import (
    create_llama2_actor,
    create_llama2_actor_critic,
    create_llama2_critic,
)
from PolyRL.models.lstm import (
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from PolyRL.models.utils import adapt_state_dict
from PolyRL.vocabulary.mytokenizers import (
    AsciiSMILESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
)


def extract(path):
    """Extract tarfile if it exists."""
    if not path.exists():
        tar_path = path.with_suffix(".tar.gz")
        if tar_path.exists():
            logging.info("Extracting model checkpoint...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=tar_path.parent)
                return path
        else:
            raise FileNotFoundError(f"File {path} not found.")
    else:
        return path


# extracting big model files only if they are not already extracted
if not (resources.files("PolyRL.priors") / "gpt2_enamine_real.ckpt").exists():
    extract(resources.files("PolyRL.priors") / "gpt2_enamine_real.ckpt")


models = {
    "gru": (
        create_gru_actor,
        create_gru_critic,
        create_gru_actor_critic,
        resources.files("PolyRL.priors") / "enamine_real_vocabulary.txt",
        resources.files("PolyRL.priors") / "grutest2.pt",
        SMILESTokenizerChEMBL(),
    ),
    "lstm": (
        create_lstm_actor,
        create_lstm_critic,
        create_lstm_actor_critic,
        resources.files("PolyRL.priors") / "enamine_real_vocabulary.txt",
        resources.files("PolyRL.priors") / "lstmB.pt",
        SMILESTokenizerEnamine(),
    ),
    "gpt2": (
        create_gpt2_actor,
        create_gpt2_critic,
        create_gpt2_actor_critic,
        resources.files("PolyRL.priors") / "enamine_real_vocabulary.txt",
        resources.files("PolyRL.priors") / "gptfinetuned_A.pth",
        SMILESTokenizerEnamine(),
    ),
    "llama2": (
        create_llama2_actor,
        create_llama2_critic,
        create_llama2_actor_critic,
        #resources.files("PolyRL.priors") / "ascii.pt",
        resources.files("PolyRL.priors") / "enamine_real_vocabulary.txt",
        resources.files("PolyRL.priors") / "llama2_finetuned_A.pth",
        SMILESTokenizerEnamine(),
        #AsciiSMILESTokenizer(),
    ),
}


def register_model(name, factory):
    """Register a model factory.

    The factory can be a function or a string in the form "module.factory".
    Running the factory should return a tuple with the following elements:
    - create_actor: a function that creates the actor model
    - create_critic: a function that creates the critic model (optional, otherwise use None)
    - create_actor_critic: a function that creates the actor-critic model (optional, otherwise use None)
    - vocabulary: a path to the vocabulary file
    - checkpoint: a path to the model checkpoint
    - tokenizer: a tokenizer instance (optional, otherwise use None)

    For more details, see the tutorial in PolyRL-open/tutorials/adding_custom_model.md.
    """
    if isinstance(factory, str):
        m, f = factory.rsplit(".", 1)
        factory = getattr(import_module(m), f)
    models[name] = factory
