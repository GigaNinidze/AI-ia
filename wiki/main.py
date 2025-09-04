import os
import sys
import xml.etree.ElementTree as ET

def main(xml_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Namespace from the dump header
    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

    context = ET.iterparse(xml_file, events=("end",))
    count = 0
    saved = 0
    skipped = 0

    for event, elem in context:
        if elem.tag.endswith("page"):
            title = elem.find("mw:title", ns).text
            ns_val = elem.find("mw:ns", ns).text
            text_elem = elem.find("mw:revision/mw:text", ns)
            text = text_elem.text if text_elem is not None else ""

            short_title = (title[:50] + "...") if len(title) > 50 else title
            print(f"[PAGE {count}] {short_title}")

            if ns_val != "0":
                print(f"   └─ SKIP: namespace={ns_val}")
                skipped += 1
            elif not text.strip():
                print(f"   └─ SKIP: empty article")
                skipped += 1
            else:
                safe_title = title.replace("/", "_").replace(" ", "_")
                out_path = os.path.join(out_dir, f"{safe_title}.txt")

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

                saved += 1
                print(f"   └─ SAVED as {out_path}")

                if saved % 100 == 0:
                    print(f"[SAVE] Total saved so far: {saved}")

            count += 1
            elem.clear()

    print("\n========== SUMMARY ==========")
    print(f"Processed pages: {count}")
    print(f"Saved articles : {saved}")
    print(f"Skipped        : {skipped}")
    print(f"Output dir     : {out_dir}")
    print("=============================\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <dump.xml> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])