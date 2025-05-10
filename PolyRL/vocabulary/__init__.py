from PolyRL.vocabulary.base import BaseTokenizer, BaseVocabulary
from PolyRL.vocabulary.mytokenizers import (
    AISTokenizer,
    AsciiSMILESTokenizer,
    DeepSMILESTokenizer,
    SAFETokenizer,
    SELFIESTokenizer,
    SMILESTokenizerChEMBL,
    SMILESTokenizerEnamine,
    SMILESTokenizerGuacaMol,
)
from PolyRL.vocabulary.vocabulary import Vocabulary

tokenizer_options = {
    "AISTokenizer": AISTokenizer,
    "DeepSMILESTokenizer": DeepSMILESTokenizer,
    "SAFETokenizer": SAFETokenizer,
    "SELFIESTokenizer": SELFIESTokenizer,
    "SMILESTokenizerChEMBL": SMILESTokenizerChEMBL,
    "SMILESTokenizerEnamine": SMILESTokenizerEnamine,
    "SMILESTokenizerGuacaMol": SMILESTokenizerGuacaMol,
    "AsciiSMILESTokenizer": AsciiSMILESTokenizer,
}
