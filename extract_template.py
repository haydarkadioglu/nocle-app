import zipfile, re
p = r'docs\2209-a_sonuc_raporu_formati.docx'
with zipfile.ZipFile(p) as z:
    xml = z.read('word/document.xml').decode('utf-8', errors='ignore')
    text = re.sub(r'<[^>]+>', ' ', xml)
    text = re.sub(r'\s+', ' ', text).strip()
    print(text[:20000])
