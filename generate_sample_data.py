"""
Generates data/sample_reviews.csv — a synthetic but realistic set of product/service
reviews spanning several domains (electronics, restaurant, hotel, streaming/movies,
clothing). Built from templated phrase pools so the topics that emerge from LDA/NMF
line up with these domains — useful for demonstrating and sanity-checking the pipeline.

Replace this file entirely with your own CSV (a single 'review' text column, optionally
a 'rating' column) for real-world use — main.py works with any reviews CSV.
"""
import random
import csv

random.seed(42)

electronics = [
    "The battery life on this phone is incredible, lasts almost two full days on a single charge.",
    "Screen resolution is stunning but the battery drains way too fast for daily use.",
    "Camera quality in low light is disappointing, photos come out grainy and blurry.",
    "Fast charging is a game changer, went from 10% to 80% in under thirty minutes.",
    "The laptop overheats badly after an hour of gaming, fan noise is very loud.",
    "Build quality feels premium, the aluminum body is sturdy and looks great.",
    "Bluetooth keeps disconnecting randomly, very frustrating during calls.",
    "Storage fills up quickly, wish it came with more internal memory options.",
    "The display has beautiful colors and the refresh rate makes scrolling super smooth.",
    "Customer support was unhelpful when I asked about a defective charging port.",
    "Software updates have made the device noticeably slower than when I bought it.",
    "Speakers are surprisingly loud and clear for such a compact device.",
    "The processor handles multitasking effortlessly, apps open instantly.",
    "Packaging was damaged on arrival and the charger was missing from the box.",
    "Fingerprint sensor is unreliable, often takes several tries to unlock.",
    "Great value for money, performance rivals phones twice the price.",
    "The webcam quality is poor, video calls look pixelated and dark.",
    "Keyboard feels mushy and the trackpad is not very responsive.",
    "Wireless charging works flawlessly with my existing charging pad.",
    "The warranty process was smooth and they replaced my unit within a week.",
]

restaurant = [
    "The pasta was cooked perfectly and the sauce had a rich, homemade flavor.",
    "Service was painfully slow, we waited over forty minutes just for appetizers.",
    "Portion sizes were tiny for the price, left the table still hungry.",
    "Our waiter was attentive and made great recommendations from the menu.",
    "The steak was overcooked and quite tough, definitely not worth the price.",
    "Ambiance is cozy and romantic, perfect for a date night dinner.",
    "Food arrived cold and had to be sent back to the kitchen twice.",
    "The dessert menu is fantastic, the chocolate lava cake was heavenly.",
    "Restaurant was extremely noisy, hard to have a conversation at our table.",
    "Fresh ingredients really shine through in every dish we ordered.",
    "The reservation system is a mess, we had a booking but still had to wait.",
    "Chef's tasting menu was an incredible culinary experience worth every penny.",
    "Portions were generous and the flavors were bold and well balanced.",
    "The bread basket was stale and the butter tasted off.",
    "Staff were friendly and checked on us regularly throughout the meal.",
    "Prices have gone up significantly but the quality has not improved.",
    "Cocktails were creative and beautifully presented by the bartender.",
    "The kitchen clearly cares about presentation, every plate looked like art.",
    "Table was sticky and the silverware looked like it hadn't been cleaned.",
    "Vegetarian options were limited and mostly just salads.",
]

hotel = [
    "The room was spotless and the bed was one of the most comfortable I've slept in.",
    "Front desk staff were rude and check-in took nearly an hour.",
    "Amazing view of the city skyline right from our balcony window.",
    "The air conditioning was broken and maintenance never showed up.",
    "Breakfast buffet had a great variety and everything was fresh.",
    "Room smelled musty and the carpet looked like it hadn't been vacuumed in weeks.",
    "Location is perfect, walking distance to all the major attractions.",
    "The pool area was closed for renovation with no warning on the website.",
    "Housekeeping did a fantastic job, room was tidy every single day.",
    "Wifi kept cutting out constantly, made it impossible to work remotely.",
    "The concierge helped us book tours and was incredibly knowledgeable.",
    "Thin walls meant we could hear everything from the room next door.",
    "Spa services were relaxing and the staff were true professionals.",
    "Elevator was out of service for our entire three night stay.",
    "Room service arrived quickly and the food was surprisingly good for a hotel.",
    "Parking was a nightmare and they charged an outrageous daily fee.",
    "Beautiful lobby with elegant decor that set a great first impression.",
    "Check-out process was quick and the final bill matched what we expected.",
    "Mini fridge in the room wasn't cooling properly, drinks stayed warm.",
    "Staff went above and beyond to accommodate our late arrival.",
]

streaming = [
    "The plot twist in the finale completely caught me off guard, brilliant writing.",
    "Acting felt wooden and the dialogue was cringeworthy throughout.",
    "Cinematography is gorgeous, every scene looks like a painting.",
    "Pacing was way too slow in the middle episodes, almost gave up watching.",
    "The soundtrack perfectly matches the mood of every scene.",
    "Character development felt rushed, motivations didn't make sense.",
    "Streaming quality kept buffering even with a strong internet connection.",
    "The ending left too many plot threads unresolved for a satisfying conclusion.",
    "Lead actor delivered a powerful, award-worthy performance.",
    "Special effects looked cheap compared to other shows in the genre.",
    "Binge-watched the whole season in one weekend, absolutely addictive story.",
    "Subtitles were out of sync for most of the second half of the movie.",
    "The humor lands perfectly, laughed out loud multiple times.",
    "Storyline felt like a copy of other popular shows with nothing original.",
    "App interface is clean and recommendations are actually relevant to my taste.",
    "Too many ads interrupted the movie even with a paid subscription.",
    "The chemistry between the two leads makes the romance believable.",
    "Season two feels like a step down from the excellent first season.",
    "Documentary presented complex topics in a clear and engaging way.",
    "Video quality drops to standard definition randomly during playback.",
]

clothing = [
    "Fabric feels premium and the stitching is very well done for the price.",
    "Sizing runs small, had to return it for a size up.",
    "Color faded noticeably after just a couple of washes.",
    "Perfect fit right out of the box, exactly as described on the site.",
    "Material is scratchy and uncomfortable against the skin.",
    "Shipping was fast and the packaging kept everything wrinkle free.",
    "Stitching came undone at the seams after only a few wears.",
    "Love the design, gets compliments every time I wear it.",
    "Return process was a hassle and customer service was slow to respond.",
    "Great quality denim that only gets better with every wash.",
    "The zipper broke within the first week of light use.",
    "True to size and the fabric breathes well even in warm weather.",
    "Photos on the website are misleading, actual color looks very different.",
    "Extremely comfortable for everyday wear, my new favorite jacket.",
    "Buttons started falling off after minimal use, poor quality control.",
]

pools = {
    "electronics": electronics,
    "restaurant": restaurant,
    "hotel": hotel,
    "streaming": streaming,
    "clothing": clothing,
}

rows = []
review_id = 1
for domain, pool in pools.items():
    # include each hand-written review once, then some lightly resampled repeats
    # with random pairing to build up dataset size and topic density
    reviews_for_domain = list(pool)
    random.shuffle(reviews_for_domain)
    for text in reviews_for_domain:
        rows.append((review_id, domain, text))
        review_id += 1
    # add combined two-sentence reviews for extra volume and realism
    for _ in range(15):
        a, b = random.sample(pool, 2)
        rows.append((review_id, domain, a + " " + b))
        review_id += 1

random.shuffle(rows)

out_path = "/home/claude/topic_modeling_reviews/data/sample_reviews.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["review_id", "true_domain", "review"])
    for review_id, domain, text in rows:
        writer.writerow([review_id, domain, text])

print(f"Wrote {len(rows)} reviews to {out_path}")
