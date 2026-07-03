# Known Faces Directory
#
# Organize your enrollment images as follows:
#
#   data/known_faces/
#   ├── Alice/
#   │   ├── alice_front.jpg
#   │   └── alice_side.jpg
#   └── Bob/
#       └── bob_photo.jpg
#
# Then run:
#   python scripts/enroll_face.py --from-dir data/known_faces/
#
# Alternatively, enroll via webcam:
#   python scripts/enroll_face.py --name "Alice" --camera 0
