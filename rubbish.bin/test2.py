from third import *


img = FShape(FChessboard(32), Meta([512, 512], float, []))

db = FDouble(img)
db.src = FLaplacianBlur(db)
init = RFCopy(db, img)

init.run()
for gui in Canvas(db):
    db.run()
