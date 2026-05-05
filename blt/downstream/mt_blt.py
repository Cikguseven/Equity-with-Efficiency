from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import sys
from pathlib import Path
from typing import Optional

import sacrebleu
import torch
import torch._dynamo
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pythainlp.tokenize import word_tokenize as thai_word_tokenize
from indicnlp.tokenize import indic_tokenize
from khmernltk import word_tokenize as khmer_word_tokenize
from laonlp.tokenize import word_tokenize as lo_word_tokenize
from myTokenize import WordTokenizer as myWordTokenizer

import os
os.environ["HF_DATASETS_CACHE"] = "/localhome/user/.hf"
os.environ["HF_HOME"]           = "/localhome/user/.hf"
os.environ["TMPDIR"]            = "/localhome/user/.hf"


# ---------------------------------------------------------------------------
# 0. DISTRIBUTED HELPERS
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    """True on rank 0, or when distributed is not initialized (single-GPU)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def dist_barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# 1. FLORES+ LOCAL FILE MAPPING
# ---------------------------------------------------------------------------

FLORES_FILENAME = {
    "en": "eng_Latn",
    "id": "ind_Latn",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
    "ms": "zsm_Latn",
    "my": "mya_Mymr",
    "ta": "tam_Taml",
    "th": "tha_Thai",
    "tl": "fil_Latn",
    "vi": "vie_Latn",
    "zh": "cmn_Hans",
}

LANG_EN_NAMES = {
    "id": "Indonesian", "km": "Khmer",   "lo": "Lao",
    "ms": "Malay",      "my": "Burmese", "ta": "Tamil",
    "th": "Thai",       "tl": "Filipino","vi": "Vietnamese",
    "zh": "Chinese",    "en": "English",
}

SEA_LANGS = ["id", "km", "lo", "ms", "my", "ta", "th", "tl", "vi", "zh"]

_SHOT_EN = [
    "The War of Spanish Succession marked the first war whose central issue was the balance of power.",
    "Many entire nations are completely fluent in English, and in even more you can expect a limited knowledge - especially among younger people.",
    "Fewer than a thousand cases have ever been reported in humans, but some of them have been fatal.",
    "In the last 3 months, over 80 arrestees were released from the Central Booking facility without being formally charged.",
    "The focus of this mindset is speed, logic and accuracy, also identification of facts, reapplying existing techniques, gathering information.",
]

_SHOT_TGT = {
    "id": [
        "Perang Penerus Spanyol menandai perang pertama dengan isu sentral berupa keseimbangan kekuatan.",
        "Banyak negara yang sepenuhnya fasih dalam bahasa Inggris, bahkan lebih banyak yang memiliki pengetahuan terbatas dari yang dapat Anda perkirakan - terutama di kalangan anak muda.",
        "Kurang dari seribu kasus pernah dilaporkan terjadi pada manusia, tetapi beberapa di antaranya merupakan kasus yang fatal.",
        "Dalam 3 bulan terakhir, lebih dari 80 tahanan dibebaskan dari fasilitas Central Booking tanpa dikenakan biaya secara resmi.",
        "Fokus dari pola pikir ini adalah kecepatan, logika, dan ketepatan, juga identifikasi fakta, penerapan kembali teknik yang ada, pengumpulan informasi.",
    ],
    "km": [
        "សង្គ្រាមដណ្តើមអំណាចគ្នារបស់ប្រទេសអេស្ប៉ាញ គឺជាសង្គ្រាមដំបូងគេ ដែល​បញ្ហាស្នូលនៃ​សង្គ្រាម​នេះ​ គឺ​តុល្យភាព​អំណាច។",
        "មានប្រជា​ជាតិ​ជាច្រើន​ដែល​ចេះ​ប្រើ​ភាសាអង់គ្លេស​បាន​ល្អ​ ហើយ​ជាង​នេះ​ទៅ​ទៀត​ ​អ្នក​អាចរំពឹង​ថា ពួកគេមានចំណេះមាន​កម្រិត​ ជាពិសេស​ក្នុង​ចំណោមមនុស្សវ័យក្មេង។",
        "បានរាយការណ៍ថា​មាន​តិចជាងមួយពាន់ករណី​បានកើតឡើង​ជាមួយនឹងមនុស្ស ប៉ុន្តែពួកគេខ្លះបានស្លាប់បាត់ទៅហើយ។",
        "ក្នុងរយៈពេល 3 ខែ ចុងក្រោយ ជន​ត្រូវ​ចាប់ខ្លួន​ជាង 80 នាក់ ត្រូវ​បាន​ដោះលែង​ចេញ​ពី​មជ្ឈមណ្ឌល​ចុះ​បញ្ជី​កណ្តាល​ដោយ​មិន​បាន​ចោទ​ប្រកាន់​ជាផ្លូវការ​ឡើយ។",
        "ការផ្តោតលើផ្នត់គំនិតនេះ គឺល្បឿន តក្កវិជ្ជា និងភាពត្រឹមត្រូវ ព្រមទាំងការកំណត់ការពិត ការអនុវត្តបច្ចេកទេសដែលមានស្រាប់ឡើងវិញ ការប្រមូលព័ត៌មាន។",
    ],
    "lo": [
        "ສົງຄາມແຫ່ງຄວາມສຳເລັດຂອງແອັດສະປາຍໄດ້ເຮັດສົງຄາມຄັ້ງທຳອິດເຊິ່ງບັນຫາໃຈກາງແມ່ນຄວາມສົມດຸນຂອງອຳນາດ.",
        "ທັງໝົດປະເທດຂອງຫຼາຍໆປະເທດແມ່ນມີຄວາມຄ່ອງແຄ້ວໃນການໃຊ້ພາສາອັງກິດ ແລະ ມັນຫຼາຍກວ່າທີ່ທ່ານຈະຄາດຫວັງເຖິງຄວາມຮູ້ທີ່ຈຳກັດ - ໂດຍສະເພາະໃນກຸ່ມໄວໜຸ່ມ.",
        "ມີຫນ້ອຍກວ່າພັນກໍລະນີທີ່ເຄີຍຖືກລາຍງານມາກ່ຽວກັບມະນຸດ, ແຕ່ວ່າມີບາງກໍລະນີກໍເປັນໂຣກຮ້າຍແຮງ.",
        "ໃນ 3 ເດືອນຜ່ານມາ, ຜູ້ຈັບກຸມຫຼາຍກວ່າ 80 ຄົນໄດ້ຖືກປ່ອຍຕົວອອກຈາກສູນກັກກັນໂດຍບໍ່ມີຂໍ້ກ່າວຫາຢ່າງເປັນທາງການ.",
        "ຈຸດລວມສຸມຂອງທັດສະນະຄະຕິນີ້ແມ່ນ ຄວາມໄວ, ໂລຊິກ ແລະ ຄວາມຖືກຕ້ອງແມ່ນຢຳ, ພ້ອມທັງການກຳນົດຂໍ້ເທັດຈິງ, ການນຳເຕັກນິກທີ່ມີຢູ່ມາໝູນໃຊ້ໃໝ່, ການລວບລວມຂໍ້ມູນ.",
    ],
    "ms": [
        "Perang Pewarisan Sepanyol menandakan perang pertama yang mana keseimbangan kuasa merupakan persoalan pertama.",
        "Banyak negara petah dalam bahasa Inggeris, dan dalam lebih banyak negara anda boleh menjangkakan pengetahuan yang berbatas - terutamanya dalam khalayak orang muda.",
        "Kurang daripada seribu kes yang pernah dilaporkan pada manusia, tetapi beberapa antaranya membawa maut.",
        "Dalam 3 bukan yang lepas, lebih 80 orang yang ditangkap telah dibebaskan daripada fasiliti Central Booking tanpa dicaj secara formal.",
        "Fokus untuk set pemikiran ini adalah kelajuan, logik, dan ketepatan, serta pengenalpastian fakta, menggunakan semua teknik yang ada, serta mengumpul maklumat.",
    ],
    "my": [
        "စပိန်တို့၏စစ်ပွဲအောင်မြင်မှုသည် အဓိကပြဿနာမှာ အာဏာ၏ဟန်ချက်ညီမှုဖြစ်ခဲ့သော ပထမအကြိမ်စစ်ပွဲကို အမှတ်အသားဖြစ်စေခဲ့ပါသည်။",
        "နိုင်ငံသားအများစုက အင်္ဂလိပ်လို ကောင်းစွာ ပြောနိုင်ပြီး အထူးသဖြင့် လူငယ်များသည် သင်ထင်ထားသည်ထက်ကို ပို၍ ဗဟုသုတ ရှိကြပါသည်။",
        "လူတစ်ထောင်ခန့် ဖြစ်ပွားကြောင်း အစီရင်ခံထားသော်လည်း ၎င်းတို့အနက်အချို့မှာ သေစေနိုင်ပါသည်။",
        "လွန်ခဲ့သော ၃ လတွင်၊ ဖမ်းဆီးခံရသူပေါင်း ၈၀ ကျော်ကို တရားဝင်တရားစွဲဆိုခြင်းမရှိဘဲ Central Booking အဆောက်အအုံမှ လွှတ်ပေးခဲ့ပါသည်။",
        "ဤစိတ်သဘောထား၏ အာရုံစိုက်မှုမှာ လက်ရှိနည်းလမ်းများကို ပြန်လည်အသုံးချရင်း အချက်အလက်များ စုဆောင်းကာ မြန်နှုန်း၊ ယုတ္တိဗေဒနှင့် တိကျမှုတို့အပြင် အချက်လက်များ ဖော်ထုတ်ခြင်း ဖြစ်သည်။",
    ],
    "ta": [
        "அதிகார சமநிலைதான் ஸ்பானிஷ் வாரிசுப் போருக்கான முதல் போரின் முக்கிய பிரச்சனையாக இருந்தது.",
        "பல தேசங்கள் முழுவதுமே ஆங்கிலத்தில் சரளமாக உள்ளன மற்றும் மேலும் பலவற்றில் குறைந்த அளவு ஆங்கில அறிவை, குறிப்பாக இளைஞர்களிடத்தில், எதிர் பார்க்கலாம்.",
        "மனிதர்களில் ஆயிரத்திற்கும் குறைவான வழக்குகள் அறிவிக்கப்பட்டுள்ளன, ஆனால், அவற்றில் சில உயிரைப் போக்கியவை.",
        "கடந்த 3 மாதங்களில், 80-க்கும் மேற்பட்ட கைது செய்யப்பட்டவர்கள், முறையாகக் குற்றஞ்சாட்டு தாக்கல் செய்யப்படாமல், மத்திய முன்பதிவு நிலையத்திலிருந்து விடுவிக்கப்பட்டனர்.",
        "இந்த மனப்பாங்கின் கவனம், வேகம், லாஜிக் மற்றும் துல்லியம் ஆகியவை, மேலும் உண்மைகளின் கண்டுபிடிப்பு, இருக்கும் உத்திகளின் மறு பயன்பாடு, தகவல் சேகரித்தல்.",
    ],
    "th": [
        "สงครามสืบราชบัลลังก์สเปนนับเป็นสงครามครั้งแรกซึ่งปัญหาหลักคือเรื่องดุลอำนาจ",
        "หลายประเทศที่ประชากรทุกคนใช้ภาษาอังกฤษได้อย่างคล่องแคล่วทั่วทั้งประเทศ และยิ่งไปกว่านั้นคุณคาดการณ์ได้เลยว่าจะพบกับคนที่มีความรู้จำกัด โดยเฉพาะในกลุ่มประชากรที่มีอายุน้อยกว่า",
        "มีรายงานผู้ป่วยน้อยกว่าหนึ่งพันรายในมนุษย์ แต่บางรายมีความรุนแรงถึงขั้นทำให้เสียชีวิตได้",
        "ในช่วง 3 เดือนที่ผ่านมา ผู้ถูกจับกุมกว่า 80 รายได้รับการปล่อยตัวจากศูนย์บันทึกข้อหากลางโดยไม่ได้ถูกตั้งข้อหาอย่างเป็นทางการ",
        "หลักสำคัญของวิธีคิดนี้คือ ความรวดเร็ว ตรรกะ และความแม่นยำ รวมทั้งการบ่งบอกข้อเท็จจริง การนำเทคนิคที่มีอยู่แล้วมาใช้ซ้ำ และการรวบรวมข้อมูล",
    ],
    "tl": [
        "Ang War of Spanish Succession (Digmaan para sa Halilinan sa Espanya) ang naging unang digmaan na pangunahing naglalayon na makamit ang balanse sa kapangyarihan.",
        "Maraming mga bansa ang matatas sa wikang Ingles, at asahan mo ang limitadong kaalaman - lalo na sa mga mas bata.",
        "Mas mababa pa sa ilang libong kaso ang naiulat sa mga tao, ngunit ang ilan sa kanila ay nakamatay.",
        "Sa huling 3 buwan, mahigit sa 80 taong naaresto ang pinalaya mula sa pasilidad ng Himpilan ng Pulisya nang hindi pormal na kinakasuhan.",
        "Ang tuon ng ganitong pag-iisip ay ang bilis, lohika at kawastuan, pati na rin ang pagkilala sa mga katotohanan, muling paggamit sa mga umiiral nang pamamaraan, pagtitipon ng impormasyon.",
    ],
    "vi": [
        "Chiến tranh Kế vị Tây Ban Nha đã đánh dấu chiến tranh đầu tiên mà vấn đề trọng tâm là sự cân bằng quyền lực.",
        "Nhiều quốc gia hoàn toàn thông thạo tiếng Anh, và ở nhiều quốc gia khác người dân cũng hiểu biết phần nào - nhất là trong số những người trẻ tuổi.",
        "Chỉ có chưa tới một ngàn ca bệnh ở người được báo cáo, nhưng một số ca đã dẫn đến tử vong.",
        "Trong vòng 3 tháng qua, đã có hơn 80 người bị bắt được thả ra khỏi trụ sở của Central Booking và không bị buộc tội chính thức.",
        "Sự tập trung tâm trí là tốc độ, sự hợp lý và tính chính xác, cũng như sự xác định thực tế, áp dụng lại các kỹ thuật có sẵn, thu thập thông tin.",
    ],
    "zh": [
        "西班牙继承权之战标志着第一场以权力平衡为核心问题的战争。",
        "在许多国家/地区，全国的人都能说一口流利的英语，而在更多的国家/地区，人们对英语也略知一二，尤其是年轻人。",
        "据报道，人类感染禽流感的病例不到 一千例，但其中有一些病例是致命的。",
        "在过去的 3 个月里，超过 80 名被捕者在没有被正式起诉的情况下从中央拘留所释放。",
        "这种思维模式看重的是速度、逻辑和准确性，还有甄别事实、对现有技术的重新应用和信息收集。",
    ],
}


def load_flores_local(src_lang: str, tgt_lang: str, flores_dir: str | Path) -> tuple[list[str], list[str]]:
    flores_dir = Path(flores_dir)
    src_file = flores_dir / f"{FLORES_FILENAME[src_lang]}.devtest"
    tgt_file = flores_dir / f"{FLORES_FILENAME[tgt_lang]}.devtest"

    if not src_file.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {src_file}")
    if not tgt_file.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {tgt_file}")

    src_sents = [s for s in src_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    tgt_sents = [s for s in tgt_file.read_text(encoding="utf-8").splitlines() if s.strip()]

    if len(src_sents) != len(tgt_sents):
        raise ValueError(
            f"Line count mismatch: {src_file} ({len(src_sents)}) vs {tgt_file} ({len(tgt_sents)})"
        )
    return src_sents, tgt_sents


# ---------------------------------------------------------------------------
# 2. PROMPT BUILDING  (X-ALMA format)
# ---------------------------------------------------------------------------
# X-ALMA prompt: "Translate this from {Src} to {Tgt}:\n{Src}: <text>\n{Tgt}:"
# Reference: https://github.com/fe1ixxu/ALMA

def build_prompt(src_lang: str, tgt_lang: str, test_src: str) -> str:
    """Return a bare X-ALMA prompt (before chat-template wrapping)."""
    src_name = LANG_EN_NAMES[src_lang]
    tgt_name = LANG_EN_NAMES[tgt_lang]
    return (
        f"Translate this from {src_name} to {tgt_name}:\n"
        f"{src_name}: {test_src}\n"
        f"{tgt_name}:"
    )


def build_fewshot_prompt(
    src_lang: str,
    tgt_lang: str,
    test_src: str,
    n_shots: int = 5,
) -> str:
    src_name = LANG_EN_NAMES[src_lang]
    tgt_name = LANG_EN_NAMES[tgt_lang]

    parts: list[str] = []

    if n_shots > 0 and tgt_lang in _SHOT_TGT:
        first_s = _SHOT_EN[0]
        first_t = _SHOT_TGT[tgt_lang][0]
        parts.append(
            f"Translate this from {src_name} to {tgt_name}:\n"
            f"{src_name}: {first_s}\n"
            f"{tgt_name}: {first_t}"
        )
        for s, t in zip(_SHOT_EN[1:n_shots], _SHOT_TGT[tgt_lang][1:n_shots]):
            parts.append(
                f"{src_name}: {s}\n"
                f"{tgt_name}: {t}"
            )
        parts.append(
            f"{src_name}: {test_src}\n"
            f"{tgt_name}:"
        )
    else:
        parts.append(
            f"Translate this from {src_name} to {tgt_name}:\n"
            f"{src_name}: {test_src}\n"
            f"{tgt_name}:"
        )

    return "\n".join(parts)


def apply_chat_template(
    prompts: list[str],
    tokenizer,
) -> list[str]:
    """
    Wrap each bare prompt in the model's chat template (required by X-ALMA
    instruction-tuned models). Falls back to returning bare prompts if the
    tokenizer has no chat_template (e.g. BLT or base HF models).
    """
    has_template = (
        getattr(tokenizer, "chat_template", None) is not None
        or getattr(tokenizer, "default_chat_template", None) is not None
    )

    if not has_template:
        if is_main_process():
            print("[WARN] tokenizer.chat_template is not set — using bare prompts (no chat wrapping).")
        return prompts

    wrapped = []
    for p in prompts:
        chat = [{"role": "user", "content": p}]
        wrapped.append(
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return wrapped


# ---------------------------------------------------------------------------
# 3. LANGUAGE-SPECIFIC PRE-TOKENIZATION
# ---------------------------------------------------------------------------

PRETOK_LANGS = {"th", "km", "lo", "my", "ta"}
TIMEOUT_SEC = 1
HAS_SIGALRM = hasattr(signal, "SIGALRM")


class TimeoutError_(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError_()


if HAS_SIGALRM:
    signal.signal(signal.SIGALRM, _alarm_handler)


def pretokenize_line(line: str, lang: str) -> str:
    lang = lang.lower()
    if lang == "th":
        return " ".join(thai_word_tokenize(line, engine="newmm"))
    if lang == "ta":
        return " ".join(indic_tokenize.trivial_tokenize(line, lang="ta"))
    if lang == "km":
        return " ".join(khmer_word_tokenize(line, return_tokens=True))
    if lang == "lo":
        if not HAS_SIGALRM:
            try:
                return " ".join(lo_word_tokenize(line))
            except Exception:
                return line
        signal.alarm(TIMEOUT_SEC)
        try:
            return " ".join(lo_word_tokenize(line))
        except TimeoutError_:
            return line
        except Exception:
            return line
        finally:
            signal.alarm(0)
    if lang == "my":
        return " ".join(myWordTokenizer().tokenize(line))
    raise ValueError(f"pretokenize_line called on non-pretok lang: {lang}")


def pretokenize_corpus(lines: list[str], lang: str) -> list[str]:
    return [pretokenize_line(line, lang) for line in lines]


# ---------------------------------------------------------------------------
# 4. METRICS: BLEU + chrF++
# ---------------------------------------------------------------------------

def build_bleu_metric(tgt_lang: str, effective_order: bool) -> sacrebleu.BLEU:
    if tgt_lang == "en":
        return sacrebleu.BLEU(tokenize="13a", effective_order=effective_order)
    if tgt_lang == "zh":
        return sacrebleu.BLEU(tokenize="zh", effective_order=effective_order)
    if tgt_lang in PRETOK_LANGS:
        return sacrebleu.BLEU(tokenize="none", effective_order=effective_order)
    return sacrebleu.BLEU(tokenize="intl", effective_order=effective_order)


def build_chrf_metric() -> sacrebleu.CHRF:
    return sacrebleu.CHRF()


def compute_bleu(hyps_tok: list[str], refs_tok: list[str], tgt_lang: str) -> float:
    metric = build_bleu_metric(tgt_lang, False)
    return metric.corpus_score(hyps_tok, [refs_tok]).score


def compute_chrf(hyps_raw: list[str], refs_raw: list[str]) -> float:
    metric = build_chrf_metric()
    return metric.corpus_score(hyps_raw, [refs_raw]).score


def compute_sentence_bleu(hyps_tok: list[str], refs_tok: list[str], tgt_lang: str) -> list[float]:
    metric = build_bleu_metric(tgt_lang, True)
    return [
        round(metric.sentence_score(h, [r]).score, 6)
        for h, r in zip(hyps_tok, refs_tok)
    ]


def compute_sentence_chrf(hyps_raw: list[str], refs_raw: list[str]) -> list[float]:
    metric = build_chrf_metric()
    return [
        round(metric.sentence_score(h, [r]).score, 6)
        for h, r in zip(hyps_raw, refs_raw)
    ]


# ---------------------------------------------------------------------------
# 5. OUTPUT EXTRACTION
# ---------------------------------------------------------------------------

LABEL_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def _compile_any_label_pattern(labels: list[str]) -> re.Pattern:
    key = tuple(labels)
    if key not in LABEL_CACHE:
        LABEL_CACHE[key] = re.compile(
            r"(?im)(?:^|\s)(?:%s)\s*:\s*" % "|".join(re.escape(x) for x in labels)
        )
    return LABEL_CACHE[key]


def clean_generation_text(text) -> str:
    if isinstance(text, list):
        text = " ".join(
            " ".join(str(tok) for tok in item) if isinstance(item, list) else str(item)
            for item in text
        )
    elif not isinstance(text, str):
        text = str(text)

    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    if "Ġ" in text:
        text = text.replace(" ", "")
        text = text.replace("Ġ", " ")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return text.strip()


def strip_prompt_prefix(decoded: str, prompt: str) -> str:
    if not decoded:
        return decoded

    d = decoded.replace("\r\n", "\n").replace("\r", "\n")
    p = prompt.replace("\r\n", "\n").replace("\r", "\n")

    if d.startswith(p):
        return d[len(p):].lstrip()

    d2 = re.sub(r"\s+", " ", d).strip()
    p2 = re.sub(r"\s+", " ", p).strip()
    if d2.startswith(p2):
        return d2[len(p2):].lstrip()

    return decoded


_ALL_LANG_LABELS: list[str] = sorted(LANG_EN_NAMES.values(), key=len, reverse=True)


def extract_translation(
    raw_output,
    tgt_label: str,
    all_labels: list[str],
    verbose: bool = False,
    fallback_tgt_label: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    if isinstance(raw_output, list):
        raw_output = " ".join(
            " ".join(str(t) for t in item) if isinstance(item, list) else str(item)
            for item in raw_output
        )
    elif not isinstance(raw_output, str):
        raw_output = str(raw_output)

    text = clean_generation_text(raw_output)

    for label in ([tgt_label] + ([fallback_tgt_label] if fallback_tgt_label else [])):
        label_pat = re.compile(rf"(?im)^\s*{re.escape(label)}\s*:\s*")
        m = label_pat.match(text)
        if m is None:
            m = label_pat.search(text)
        if m:
            text = text[m.end():].lstrip()
            break

    stop_pat = _compile_any_label_pattern([x for x in all_labels if x != tgt_label])
    m2 = stop_pat.search(text)
    if m2:
        text = text[:m2.start()].strip()

    fence_idx = text.find("```")
    if fence_idx != -1:
        text = text[:fence_idx].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        out = ""
    else:
        out = lines[0]
        if not isinstance(out, str):
            out = str(out)
        out = re.sub(r"^\s*[-–—*`\"']+\s*", "", out).strip()
        out = re.sub(r"\s+", " ", out).strip()

    if verbose and is_main_process():
        if prompt is not None:
            print(prompt)
            print("---")
        print("RAW OUTPUT:")
        print(raw_output)
        print("---")
        print("EXTRACTED:")
        print(out)
        print("=" * 80)

    return out


# ---------------------------------------------------------------------------
# 6a. STANDARD HUGGINGFACE MODEL LOADING & INFERENCE
# ---------------------------------------------------------------------------

def resolve_torch_dtype(name: str):
    name = name.lower()
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported --torch_dtype: {name}")


def load_model_hf(model_path: str, torch_dtype_name: str = "auto"):
    model_path = Path(model_path).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in: {model_path}")

    cfg = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    hf_tok = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    hf_tok.padding_side = "left"

    tdtype = resolve_torch_dtype(torch_dtype_name)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=cfg,
        torch_dtype=tdtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, hf_tok, None


def _get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@torch.no_grad()
def run_inference_hf(
    model,
    hf_tok,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 16,
    max_input_length: int = 8192,
) -> list[str]:
    outputs = []
    truncation_count = 0
    input_device = _get_model_input_device(model)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        raw_tok = hf_tok(batch, add_special_tokens=False, truncation=False)
        raw_lens = [len(x) for x in raw_tok["input_ids"]]
        truncation_count += sum(x > max_input_length for x in raw_lens)

        enc = hf_tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(input_device) for k, v in enc.items()}

        gen_output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            pad_token_id=hf_tok.pad_token_id,
            eos_token_id=hf_tok.eos_token_id,
        )

        input_len = enc["input_ids"].shape[1]

        for j in range(gen_output.size(0)):
            completion_ids = gen_output[j, input_len:]
            decoded = hf_tok.decode(completion_ids, skip_special_tokens=True)
            outputs.append(decoded)

        if is_main_process() and (i // batch_size) % 10 == 0:
            print(f"  [HF] {i + len(batch)}/{len(prompts)} done")

    if truncation_count and is_main_process():
        print(f"[WARN] {truncation_count} prompt(s) exceeded max_input_length={max_input_length} and were truncated.")

    return outputs


# ---------------------------------------------------------------------------
# 6b. BLT MODEL LOADING & INFERENCE
# ---------------------------------------------------------------------------

def load_model_blt(model_path: str, entropy_model_path: str, blt_repo_path: Optional[str] = None):
    if blt_repo_path:
        repo = str(Path(blt_repo_path).expanduser().resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

    from bytelatent.distributed import DistributedArgs, setup_torch_distributed
    from bytelatent.generate import load_consolidated_model_and_tokenizer
    from bytelatent.generate_blt import generate_nocache
    from bytelatent.model.blt import ByteLatentTransformer
    from bytelatent.blt_tokenizers.blt_tokenizer import BltTokenizer

    torch._dynamo.config.suppress_errors = True

    distributed_args = DistributedArgs()
    distributed_args.configure_world()

    if not torch.distributed.is_initialized():
        # torchrun injects MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE via env://
        # No manual port assignment needed — torchrun owns the rendezvous lifecycle.
        setup_torch_distributed(distributed_args)

    if is_main_process():
        print(f"[BLT] Rank {get_rank()}/{get_world_size()} — Loading model from {model_path} ...")

    model, blt_tok, train_cfg = load_consolidated_model_and_tokenizer(model_path)

    if not isinstance(model, ByteLatentTransformer):
        raise TypeError(f"Expected ByteLatentTransformer, got {type(model)}")
    if not isinstance(blt_tok, BltTokenizer):
        raise TypeError(f"Expected BltTokenizer, got {type(blt_tok)}")

    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    patcher_args.entropy_model_checkpoint_dir = entropy_model_path

    if is_main_process():
        print(f"[BLT] Building patcher with entropy model from {entropy_model_path} ...")

    patcher = patcher_args.build()

    return model, blt_tok, patcher, generate_nocache


@torch.no_grad()
def run_inference_blt(
    model,
    blt_tok,
    patcher,
    generate_nocache_fn,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 16,
) -> list[str]:
    outputs: list[str] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch = [p if p.strip() else " " for p in batch]

        out_tokens = generate_nocache_fn(
            batch,
            model=model,
            tokenizer=blt_tok,
            patcher=patcher,
            max_gen_len=max_new_tokens,
        )

        for prompt, tokens in zip(batch, out_tokens):
            full_decoded = blt_tok.decode(tokens)
            stripped = strip_prompt_prefix(full_decoded, prompt)
            outputs.append(stripped.strip())

        if is_main_process() and (i // batch_size) % 10 == 0:
            print(f"  [BLT] {i + len(batch)}/{len(prompts)} done")

    # Ensure all ranks finish inference before any rank proceeds to I/O
    dist_barrier()

    return outputs


# ---------------------------------------------------------------------------
# 7. UNIFIED LOAD + INFERENCE
# ---------------------------------------------------------------------------

def load_model(
    tokenizer_name: str,
    model_path: str,
    entropy_model_path: str,
    torch_dtype_name: str = "auto",
    blt_repo_path: Optional[str] = None,
):
    if "blt" in tokenizer_name.lower():
        model, tokenizer, patcher, generate_nocache_fn = load_model_blt(
            model_path=model_path,
            entropy_model_path=entropy_model_path,
            blt_repo_path=blt_repo_path,
        )
        return model, tokenizer, patcher, generate_nocache_fn
    model, tokenizer, patcher = load_model_hf(model_path=model_path, torch_dtype_name=torch_dtype_name)
    return model, tokenizer, patcher, None


def run_inference(
    tokenizer_name: str,
    model,
    tokenizer,
    patcher,
    generate_nocache_fn,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    max_input_length: int,
) -> list[str]:
    if "blt" in tokenizer_name.lower():
        return run_inference_blt(
            model=model,
            blt_tok=tokenizer,
            patcher=patcher,
            generate_nocache_fn=generate_nocache_fn,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    return run_inference_hf(
        model=model,
        hf_tok=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_length=max_input_length,
    )


# ---------------------------------------------------------------------------
# 8. SINGLE DIRECTION EVALUATION
# ---------------------------------------------------------------------------

def evaluate_direction(
    tokenizer_name: str,
    model,
    tokenizer,
    patcher,
    generate_nocache_fn,
    src_lang: str,
    tgt_lang: str,
    flores_dir: str,
    max_new_tokens: int,
    batch_size: int,
    max_input_length: int,
    verbose: bool = False,
    max_sentences: Optional[int] = None,
    n_shots: int = 0,
) -> dict:
    if is_main_process():
        print(f"\n=== {src_lang} -> {tgt_lang} ===")

    src_sents, tgt_sents_raw = load_flores_local(src_lang, tgt_lang, flores_dir)

    if max_sentences is not None:
        src_sents     = src_sents[:max_sentences]
        tgt_sents_raw = tgt_sents_raw[:max_sentences]
        if is_main_process():
            print(f"  [DEBUG] Truncated to {len(src_sents)} sentences")

    if is_main_process():
        print(f"  {len(src_sents)} test sentences")

    if src_lang == "en" and n_shots > 0 and tgt_lang in _SHOT_TGT:
        bare_prompts = [
            build_fewshot_prompt(src_lang, tgt_lang, s, n_shots=n_shots)
            for s in src_sents
        ]
    else:
        bare_prompts = [build_prompt(src_lang, tgt_lang, s) for s in src_sents]

    prompts = apply_chat_template(bare_prompts, tokenizer)

    raw_out = run_inference(
        tokenizer_name=tokenizer_name,
        model=model,
        tokenizer=tokenizer,
        patcher=patcher,
        generate_nocache_fn=generate_nocache_fn,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_length=max_input_length,
    )

    # --- Rank-0 only: extraction, metrics, and return value ---
    if not is_main_process():
        return {}

    tgt_label = LANG_EN_NAMES[tgt_lang]

    hyps_raw = [
        extract_translation(
            o,
            tgt_label=tgt_label,
            all_labels=_ALL_LANG_LABELS,
            verbose=verbose,
            prompt=p,
        )
        for p, o in zip(prompts, raw_out)
    ]

    chrf = compute_chrf(hyps_raw, tgt_sents_raw)

    if tgt_lang in PRETOK_LANGS:
        print(f"  Pre-tokenizing for BLEU ({tgt_lang}) ...")
        hyps_tok = pretokenize_corpus(hyps_raw, tgt_lang)
        refs_tok = pretokenize_corpus(tgt_sents_raw, tgt_lang)
    else:
        hyps_tok = hyps_raw
        refs_tok = tgt_sents_raw

    bleu = compute_bleu(hyps_tok, refs_tok, tgt_lang)

    print(f"  Computing per-sentence scores ...")
    sent_bleu = compute_sentence_bleu(hyps_tok, refs_tok, tgt_lang)
    sent_chrf = compute_sentence_chrf(hyps_raw, tgt_sents_raw)

    empty_count = sum(1 for x in hyps_raw if not x.strip())

    print(f"  BLEU {bleu:.4f}  |  chrF++ {chrf:.4f}  |  empty {empty_count}")

    return {
        "src": src_lang,
        "tgt": tgt_lang,
        "direction": f"{src_lang}->{tgt_lang}",
        "bleu": round(bleu, 4),
        "chrf": round(chrf, 4),
        "num_sentences": len(src_sents),
        "num_empty_hypotheses": empty_count,
        "hypotheses": hyps_raw,
        "references": tgt_sents_raw,
        "prompts": bare_prompts,
        "sentence_bleu": sent_bleu,
        "sentence_chrf": sent_chrf,
    }


# ---------------------------------------------------------------------------
# 9. FULL EVALUATION LOOP
# ---------------------------------------------------------------------------

def run_full_evaluation(
    tokenizer_name: str,
    model_path: str,
    entropy_model_path: str,
    flores_dir: str,
    output_dir: str,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    directions: Optional[list[tuple[str, str]]] = None,
    max_input_length: int = 8192,
    torch_dtype_name: str = "auto",
    blt_repo_path: Optional[str] = None,
    verbose: bool = False,
    max_sentences: Optional[int] = None,
    n_shots: int = 0,
):
    if is_main_process():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer, patcher, generate_nocache_fn = load_model(
        tokenizer_name=tokenizer_name,
        model_path=model_path,
        entropy_model_path=entropy_model_path,
        torch_dtype_name=torch_dtype_name,
        blt_repo_path=blt_repo_path,
    )

    if directions is None:
        directions = [("en", tgt) for tgt in SEA_LANGS] + [(src, "en") for src in SEA_LANGS]

    results = []
    for src_lang, tgt_lang in directions:
        res = evaluate_direction(
            tokenizer_name=tokenizer_name,
            model=model,
            tokenizer=tokenizer,
            patcher=patcher,
            generate_nocache_fn=generate_nocache_fn,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            flores_dir=flores_dir,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
            verbose=verbose,
            max_sentences=max_sentences,
            n_shots=n_shots,
        )

        # All file I/O is rank-0 only
        if is_main_process():
            Path(output_dir).joinpath(f"{tokenizer_name}_{src_lang}-{tgt_lang}_hyps.txt").write_text(
                "\n".join(res["hypotheses"]),
                encoding="utf-8",
            )
            Path(output_dir).joinpath(f"{tokenizer_name}_{src_lang}-{tgt_lang}_refs.txt").write_text(
                "\n".join(res["references"]),
                encoding="utf-8",
            )

            sent_scores_path = Path(output_dir) / f"{tokenizer_name}_{src_lang}-{tgt_lang}_scores.csv"
            with open(sent_scores_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["sentence_id", "src", "tgt", "tokenizer",
                                "hypothesis", "reference",
                                "sentence_bleu", "sentence_chrf"],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "sentence_id":   idx,
                        "src":           src_lang,
                        "tgt":           tgt_lang,
                        "tokenizer":     tokenizer_name,
                        "hypothesis":    hyp,
                        "reference":     ref,
                        "sentence_bleu": sb,
                        "sentence_chrf": sc,
                    }
                    for idx, (hyp, ref, sb, sc) in enumerate(zip(
                        res["hypotheses"],
                        res["references"],
                        res["sentence_bleu"],
                        res["sentence_chrf"],
                    ))
                ])

            row = {k: v for k, v in res.items()
                   if k not in ("hypotheses", "references", "prompts",
                                "sentence_bleu", "sentence_chrf")}
            row["tokenizer"] = tokenizer_name
            results.append(row)

        # Barrier after each direction so all ranks stay in sync
        dist_barrier()

    if is_main_process():
        en_xx = [r for r in results if r["src"] == "en"]
        xx_en = [r for r in results if r["tgt"] == "en"]

        def avg(rows, key):
            return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None

        summary = {
            "tokenizer": tokenizer_name,
            "model_path": str(Path(model_path).expanduser().resolve()),
            "flores_dir": str(Path(flores_dir).expanduser().resolve()),
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "max_input_length": max_input_length,
            "torch_dtype": torch_dtype_name,
            "world_size": get_world_size(),
            "en_xx_bleu_avg": avg(en_xx, "bleu"),
            "en_xx_chrf_avg": avg(en_xx, "chrf"),
            "xx_en_bleu_avg": avg(xx_en, "bleu"),
            "xx_en_chrf_avg": avg(xx_en, "chrf"),
            "directions": [r["direction"] for r in results],
            "n_shots": n_shots,
        }

        csv_path = Path(output_dir) / f"{tokenizer_name}_results.csv"
        fieldnames = [
            "tokenizer", "direction", "src", "tgt",
            "num_sentences", "num_empty_hypotheses",
            "bleu", "chrf",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: r.get(k) for k in fieldnames} for r in results])

        Path(output_dir).joinpath(f"{tokenizer_name}_summary.json").write_text(
            json.dumps({**summary, "per_direction": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 88)
        print(f"RESULTS — {tokenizer_name}")
        print(f"  {'Direction':<12} {'BLEU':>8} {'chrF++':>8} {'Empty':>7}")
        print("  " + "-" * 44)
        for r in results:
            print(f"  {r['direction']:<12} {r['bleu']:>8.4f} {r['chrf']:>8.4f} {r['num_empty_hypotheses']:>7}")
        print(f"  EN->XX avg   {summary['en_xx_bleu_avg']:>8.4f} {summary['en_xx_chrf_avg']:>8.4f}")
        if summary["xx_en_bleu_avg"] is not None and summary["xx_en_chrf_avg"] is not None:
            print(f"  XX->EN avg   {summary['xx_en_bleu_avg']:>8.4f} {summary['xx_en_chrf_avg']:>8.4f}")
        print(f"\nSaved to: {output_dir}")

        return results, summary

    return [], {}


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FLORES+ translation eval — BLEU + chrF++ (X-ALMA prompts)"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint (HF or BLT consolidated)")
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--entropy_model_path", default=None,
                        help="[BLT only] Path to the entropy model directory.")
    parser.add_argument("--blt_repo_path", default=None,
                        help="[BLT only] Optional local path to the bytelatent repo.")
    parser.add_argument("--flores_dir",
                        default="/localhome/user/flores-plus_dev_devtest",
                        help="Local FLORES+ folder containing *.devtest files")
    parser.add_argument("--output_dir", default="/localhome/user/mt_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=4096,
                        help="HF-only prompt length cap; warnings emitted on truncation.")
    parser.add_argument("--torch_dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Subset of SEA lang codes (default: all 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw outputs and extracted translations.")
    parser.add_argument("--debug_n", type=int, default=None, metavar="N",
                        help="[Debug] Evaluate only the first N sentences per direction.")
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--en_xx_only",
        action="store_true",
        help="Only evaluate EN→XX directions; skip XX→EN.",
    )
    direction_group.add_argument(
        "--xx_en_only",
        action="store_true",
        help="Only evaluate XX→EN directions; skip EN→XX.",
    )
    parser.add_argument(
        "--n_shots", type=int, default=5,
        help="Number of few-shot examples for EN→XX (0 = zero-shot). Default: 5",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if "blt" in args.tokenizer_name.lower() and args.entropy_model_path is None:
        raise ValueError("--entropy_model_path is required when --tokenizer_name is blt")

    max_sentences: Optional[int] = None
    if args.debug_n is not None:
        if args.debug_n < 1:
            raise ValueError("--debug_n must be >= 1")
        max_sentences = args.debug_n

    directions = None
    if args.langs:
        bad = [x for x in args.langs if x not in SEA_LANGS]
        if bad:
            raise ValueError(f"Unsupported language code(s) in --langs: {bad}")
        directions = [("en", tgt) for tgt in args.langs] + [(src, "en") for src in args.langs]

    if args.en_xx_only:
        if directions is not None:
            directions = [(s, t) for s, t in directions if s == "en"]
        else:
            directions = [("en", tgt) for tgt in SEA_LANGS]

    if args.xx_en_only:
        if directions is not None:
            directions = [(s, t) for s, t in directions if t == "en"]
        else:
            directions = [(src, "en") for src in SEA_LANGS]

    run_full_evaluation(
        tokenizer_name=args.tokenizer_name,
        model_path=args.model_path,
        entropy_model_path=args.entropy_model_path or "",
        flores_dir=args.flores_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        directions=directions,
        max_input_length=args.max_input_length,
        torch_dtype_name=args.torch_dtype,
        blt_repo_path=args.blt_repo_path,
        verbose=args.verbose,
        max_sentences=max_sentences,
        n_shots=args.n_shots,
    )