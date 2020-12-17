from third import *
ti.init(ti.cc, log_level=ti.DEBUG)

ini = FShape(C.float[512, 512], FGaussDist([256, 256], 6, 8))
pos = FDouble(ini)
vel = FDouble(ini)
pos.src = FPosAdvect(pos, vel, 0.1)
vel.src = FLaplacianStep(pos, vel, 1)
init = RMerge(RFCopy(pos, ini), RFCopy(vel, FConst(0)))
step = RTimes(RMerge(pos, vel), 8)
vis = FShape(C.float(3)[512, 512], FMix(FVPack(pos, FGradient(pos)), FConst(1), 0.5, 0.5))
#frm = Field(C.int[None])
#off = FCache(FShape(C.float[None], FFunc(lambda x: 10 * ti.sin(0.1 * x), frm)))
#vis = FCache(FLike(vis, FBilerp(FRepeat(vis), FMix(FIndex(), FUniform(off), 1, 1))))
#step = RMerge(off, vis, step)

init.run()
for gui in Canvas(vis):
    #frm.core[None] = gui.frame
    step.run()
