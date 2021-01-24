import queue
import threading
import traceback



class TaichiWorkerMT:
    def __init__(self):
        self.q = queue.Queue(maxsize=4)
        self.running = True

        self.t = threading.Thread(target=self.main)
        self.t.daemon = True
        self.t.start()

    def stop(self):
        print('[TiWork] Stopping worker thread')
        try:
            if self.running:
                self.running = False
                self.q.put((lambda self: None, [None]), block=False)
        except Exception as e:
            print(e)

    def main(self):
        print('[TiWork] Worker thread started')
        while self.running:
            try:
                func, resptr = self.q.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                resptr[0] = func(self)
            except Exception:
                msg = traceback.format_exc()
                print('[TiWork] Exception while running task:\n' + msg)
                resptr[1] = msg

            self.q.task_done()

    def launch(self, func):
        resptr = [None, None]
        self.q.put((func, resptr), block=True, timeout=None)
        return resptr

    def wait_done(self):
        self.q.join()


class TaichiWorkerST:
    def __init__(self):
        pass

    def stop(self):
        pass

    def launch(self, func):
        try:
            ret = func(self)
        except Exception:
            msg = traceback.format_exc()
            print('[TiWork (ST)] Exception while running task:\n' + msg)
            return [None, msg]
        return [ret, None]

    def wait_done(self):
        pass


class TaichiWorkerLOC:
    def __init__(self, __cls=TaichiWorkerMT):
        self.__cls = __cls
        self.__core = None

    def __getattr__(self, attr):
        if self.__core is None:
            self.__core = self.__cls()

        return getattr(self.__core, attr)

    def __setattr__(self, attr, value):
        if attr.startswith('_TaichiWorkerLOC__'):
            self.__dict__[attr] = value
            return

        if self.__core is None:
            self.__core = self.__cls()
        setattr(self.__core, attr, value)


TaichiWorker = TaichiWorkerMT
