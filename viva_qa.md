# Viva: 10 common sawal aur jawab

**1. Ye project kya karta hai?**
News text lekar batata hai ki wo real hai ya fake. Text ko TF-IDF se numbers mein badalte hain aur ML classifier se label predict karte hain.

**2. TF-IDF kya hota hai?**
Term Frequency x Inverse Document Frequency. Jo shabd ek article mein zyada aaye lekin pure dataset mein kam articles mein aaye, use zyada weight milta hai. "the", "is" jaise aam shabd ka weight kam hota hai.

**3. Aapne Logistic Regression kyun use kiya?**
Text classification mein tez, simple aur achha kaam karta hai. Probability deta hai (app mein confidence dikhane ke liye) aur coefficients dekhkar samajh sakte hain ki kaunse shabd fake ya real ki taraf le jate hain.

**4. Teen models mein farq kya hai?**
Logistic Regression ek linear boundary seekhta hai. Naive Bayes shabdon ki probability par chalta hai aur shabdon ko independent maanta hai. Passive Aggressive online learning wala linear model hai jo galti par hi weights badalta hai.

**5. Accuracy ke saath precision, recall aur F1 kyun dekhe?**
Accuracy akeli kabhi dhokha deti hai. Precision batati hai ki "real" kehne wale kitne sahi the, recall batata hai ki kitne real articles pakde gaye. F1 dono ka balance hai.

**6. Confusion matrix kya batata hai?**
Kitne articles sahi aur kitne galat classify hue, aur galti kis type ki thi (fake ko real kaha ya real ko fake).

**7. Accuracy 98-99% kyun aayi? Kya model overfit hai?**
Dataset mein real news ke shuru mein `(Reuters)` likha hota hai, aur model wahi pakad leta hai. Isliye humne source words hatakar dobara test kiya aur dono score report mein dikhaye.

**8. Train aur test split kyun?**
Model ko ye dikhana hota hai ki wo naye data par kaisa chalta hai. 80% se train karte hain aur 20% par test, jo model ne pehle kabhi nahi dekha.

**9. Is model ki limitations kya hain?**
Ye sirf is dataset ki style seekhta hai. Doosre topic, language ya naye type ki fake news par kamzor ho sakta hai. Satire ya half-true khabar ko pakadna mushkil hai. Ye fact-checking ka replacement nahi hai.

**10. Aage kya improve kar sakte ho?**
BERT jaise transformer models, zyada aur alag-alag sources ka dataset, Hindi news support, aur fact-check APIs ke saath jodna.
