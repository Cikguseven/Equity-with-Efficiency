from src.myt5.myt5_tokenizer import MyT5Tokenizer

tokenizer = MyT5Tokenizer()

pre_texts = ['On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.', 'Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.']

inputs1 = tokenizer(pre_texts, padding=False, add_special_tokens=False)
# inputs2 = tokenizer2(pre_texts, padding="longest", return_tensors="pt")
# targets = tokenizer(post_texts, padding="longest", return_tensors="pt")

print(inputs1)
