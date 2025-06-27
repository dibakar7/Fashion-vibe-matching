FlickdAi/
├── app.py
├── FrameExtractor.py              # contains FrameExtractor()
├── ImageDownloader.py             # For downloading images using urls in images.csv
├── ImageEmbeddings.py             # For checking/matching images with product details in product_data.csv
├── ImageTextEmbedder.py           # For embedding images with product details in product_data.csv
├── YOLOv8Engine.py                # contains FashionItemsDetector()
├── utils/                         # optional: crop + CLIP logic
├── video_inputs/                  # place your reels/videos here
├── frames/                        # auto-generated: holds extracted frames
├── outputs/                       # results JSONs, logs, etc.
├── cropped/                       # optional: save cropped fashion items
├── images/                        # contain all downloaded images
├── product_data.csv               # csv file that contains all product details
├── images.csv                     # csv file that contains image url
└── requirements.txt               # pip install -r requirements.txt
