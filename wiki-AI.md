us1ndiso@GIGAS-LAPTOP ქართული-ენა % ls -la wiki/articles/ | head -10
total 3739376
-rw-r--r--@     1 us1ndiso  staff       85 Aug 21 01:30 შრომაჲ_და_მოღვაწებაჲ_ღირსად_ცხორებისაჲ_წმიდისა_და_ნეტარისა_მამისა_ჩვენისა_გრიგოლისი.txt
-rw-r--r--@     1 us1ndiso  staff      211 Aug 21 01:30 ვინ_არის_ჰარი_კელერმანი_და_რატომ_თქვა_ეს_საშინელებები_ჩემზე_(ფილმი).txt
-rw-r--r--@     1 us1ndiso  staff      211 Aug 21 01:30 ვინ_არის_ჰარი_კელერმანი_და_რატომ_თქვა_ეს_საშინელებები_ჩემზე?_(ფილმი).txt
-rw-r--r--@     1 us1ndiso  staff     2674 Aug 21 01:30 ვინ_არის_ჰარი_კელერმანი_და_რატომ_თქვა_ეს_საშინელებები_ჩემზე?.txt
-rw-r--r--@     1 us1ndiso  staff    14835 Aug 21 01:31 ბაჰრეინის_დაბომბვა_იტალიის_საჰაერო_ძალების_მიერ_მეორე_მსოფლიო_ომის_დროს.txt
-rw-r--r--@     1 us1ndiso  staff     7053 Aug 21 01:31 დიდ_სამამულო_ომში_საბჭოთა_კავშირის_გმირის_წოდებით_დაჯილდოებული_ქართველების_სია.txt
-rw-r--r--@     1 us1ndiso  staff       82 Aug 21 01:30 უკრაინის_სახალხო_რესპუბლიკისა_და_დასავლეთ_უკრაინის_სახალხო_რესპუბლიკის_გაერთიანების_აქტი.txt
-rw-r--r--@     1 us1ndiso  staff     8319 Aug 21 01:31 საქართველოს_სოფლის_მეურნეობის_პროდუქციაზე_ემბარგოს_დაწესება_რუსეთის_ფედერაციის_მიერ_(2006).txt
-rw-r--r--@     1 us1ndiso  staff    11131 Aug 21 01:31 აზერბაიჯანის_ლტოლვილთა_და_იძულებით_გადაადგილებულ_პირებთან_მუშაობის_სახელმწიფო_კომიტეტი.txt
us1ndiso@GIGAS-LAPTOP ქართული-ენა % du -sh wiki/articles/
1.8G    wiki/articles/
us1ndiso@GIGAS-LAPTOP ქართული-ენა % cd wiki && python analyze_corpus.py
🔍 Georgian Wikipedia Corpus Analysis
==================================================
📊 Analyzing article file sizes...
📄 Total articles: 233,926
📏 Size statistics:
   Min: 17 bytes
   Max: 723,185 bytes
   Median: 3,046 bytes
   Average: 5723 bytes
📈 Size distribution:
   Small (< 1KB): 57,152 (24.4%)
   Medium (1-10KB): 146,116 (62.5%)
   Large (> 10KB): 30,658 (13.1%)

📖 Sampling 5 random articles...

📄 Sample 1: Lasioglossum dotatum
   Length: 1,351 characters
   Preview: {{კურსივისახელი}}
{{ტაქსოდაფა
| სახელი = ''Lasioglossum dotatum''
| სურათის ფაილი =
| სურათის წარწერა =
| სურათის აღწერა =
| სამეფო = ცხოველები
| ტიპი = [[ფეხსახსრიანები]]
| კლასი = [[მწერები]]
| რიგი...

📄 Sample 2: ალმოარინი
   Length: 1,198 characters
   Preview: {{ინფოდაფა დასახლება
|სტატუსი                  = მუნიციპალიტეტი
|ქართული სახელი        = ალმოარინი
|ქვეყანა = ესპანეთი
|რეგიონის ტიპი             =  ავტონომიური გაერთიანება
|რეგიონი                  =...

📄 Sample 3: ჯეიმზ ოგლთორპი
   Length: 2,844 characters
   Preview: {{ინფოდაფა პოლიტიკოსი
| სახელი= ჯეიმზ ოგლთორპი
| მშობლიურ ენაზე= {{Lang-en|James Edward Oglethorpe}}
| სურათი= 
| სურათის ზომა=
| წარწერა=
| დაბადების სახელი=
| ფსევდონიმი= 
| დაბადების თარიღი= 
| დაბ...

📄 Sample 4: ანთაძე, დოდო
   Length: 25 characters
   Preview: #REDIRECT [[დოდო ანთაძე]]...

📄 Sample 5: NGC 2370
   Length: 1,316 characters
   Preview: {{ინფოდაფა გალაქტიკა
 | სახელი                  = NGC 2370
 | სურათი               = 
 | აღმომჩენი               = 
 | აღმოჩენის თარიღი             = [[10 ნოემბერი]], [[1864]]
 | ეპოქა                ...

🔍 Analyzing content quality (sample of 1000 articles)...
📊 Content Quality Analysis:
   Articles processed: 1,000
   Empty articles: 0
   Total characters: 3,040,542
   Total words: 308,669
   Georgian character ratio: 48.8%
   Avg chars/article: 3041
   Avg words/article: 309

📏 Article length distribution:
   Short (< 100 chars): 207
   Medium (100-1000 chars): 153
   Long (> 1000 chars): 640

🔤 Top 20 characters:
   'ი': 204,014
   'ა': 197,077
   'ე': 133,158
   'ს': 103,568
   'რ': 95,287
   'ო': 81,894
   'ლ': 73,017
   'მ': 66,277
   'ნ': 64,054
   'e': 61,350
   'დ': 53,766
   '|': 53,195
   'ბ': 50,421
   't': 47,985
   '=': 46,470
   'a': 44,789
   'r': 41,131
   '[': 40,918

📝 Top 20 words:
   '{{ინფოდაფა': 481
   'სახელი': 442
   '|ქართული': 227
   '|სტატუსი': 195
   'დასახლება': 189
   '#გადამისამართება': 170
   '|მშობლიური': 156
   'სოფელი': 74
   'მუნიციპალიტეტი': 56
   '|ქვეყანა': 55
   '|სახელი': 53
   'სურათის': 52
   'ფაილი': 49
   'ადმინისტრაციული': 45
   'ქალაქი': 39
   '#REDIRECT': 35
   '|სურათი': 33
   'სახელწოდება': 30
   '{{ტაქსოდაფა': 27
   'სურათი': 26

🎯 FINAL ESTIMATES:
   Total articles: 233,926
   Estimated total size: 678 MB
   Estimated tokens: 177815457 (assuming 4 chars/token)
   Quality: 🔴 Needs filtering

🚀 RECOMMENDATIONS:
   📈 You have MORE than enough data for excellent LLM training!
   🎯 Consider using subsets for faster iteration
   🔄 Next steps:
   1. Run: python process_wiki_corpus.py
   2. Run: python train_tokenizer.py
   3. Run: python improved_training.py
us1ndiso@GIGAS-LAPTOP wiki % 