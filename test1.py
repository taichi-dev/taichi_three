from third import *


img = FShape(FChessboard(32), Meta([512, 512], float, []))

pos = FDouble(img)
pos.src = FLaplacianStep(pos, pos, 0.1)
init = FCopy(pos, img)

init.run()
for gui in Canvas(pos):
    pos.run()
