import tina

tina.ti.init(tina.ti.gpu)

li = tina.SkyboxLighting('assets/ballroom.npy', precision=2048)
li.save('assets/ballroom.npz')
