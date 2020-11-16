from third import *
ti.init(ti.cc)


meta = C.f32[512, 512]
print(meta)
ini = FShape(FGaussDist([256, 256], 6, 8), meta)
pos = FDouble(ini)
vel = FDouble(ini)
pos.src = FPosAdvect(pos, vel, 0.1)
vel.src = FLaplacianStep(pos, vel, 1)
init = RMerge(RFCopy(pos, ini), RFCopy(vel, FConst(0)))
step = RTimes(RMerge(pos, vel), 6)
vis = FLike(pos, FMix(pos, FConst(1), 0.5, 0.5))

init.run()
for gui in Canvas(vis):
    step.run()
