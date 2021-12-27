from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
args = vars(ap.parse_args())

print("Loading encodings face...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# We could use another ML model than this
print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["encodings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()