import taichi as ti
import tina.hacker

gui = ti.GUI()

while gui.running:
    for e in gui.get_events():
        print(e.type, e.pos)
    gui.show()