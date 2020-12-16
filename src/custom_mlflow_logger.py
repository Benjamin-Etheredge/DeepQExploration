import threading, queue

import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
import time

'''
def log_metric(key, value, step):

    #client.log_batch(run.info.run_id, Metric(key=key, value=value, timestamp=0, step=step))
    #mlflow.log_metric(key=f"start_{key}", value=value, step=step)
    metric = Metric(key=key, value=value, timestamp=int(time.time() * 1000), step=step)
    metrics.append(metric)
    q.put(metric)
    #if len(metrics) > 1000:
        #run = mlflow.active_run()
        #while len(metrics):
            #client.log_batch(run.info.run_id, metrics=metrics[:1000])
            #del metrics[:1000]
'''

import multiprocessing
class Logger:
    __instance = None

    @staticmethod
    def getInstance():
        if Logger.__instance is None:
            Logger()
        return Logger.__instance
    
    def __init__(self):
        self.q = queue.Queue()
        self.client = MlflowClient()
        self.run = mlflow.active_run()
        self.metrics = []
        #self.metrics = queue.Queue()
        threading.Thread(target=self.worker, daemon=True).start()

        #self.P = multiprocessing.Process(target=self.worker)#.start()
        #self.P.start()
        Logger.__instance = self


    def worker(self):

        '''
        while True:
            metrics_batch = self.q.get()
            self.client.log_batch(self.run.info.run_id, metrics=metrics_batch)
            self.q.task_done()
            #mlflow.log_metric(key=f"worker_{metric.key}", value=metric.value, step=metric.step)
        '''


        metrics = []
        while True:
            try:
                while len(metrics) < 1000:
                    metric = self.q.get(block=True, timeout=1)
                    #mlflow.log_metric(key=metric.key, value=metric.value, step=metric.step)
                    metrics.append(metric)
                    self.q.task_done()
            except queue.Empty:
                pass

            if len(metrics) > 0:
                self.client.log_batch(self.run.info.run_id, metrics=metrics[:1000])
                del metrics[:1000]

    #def log_batch(self, metrics_batch):
        #assert(len(metrics_batch) <= 1000)
        #run = mlflow.active_run()
        #self.client.log_batch(run.info.run_id, metrics=metrics_batch)
        #del self.metrics[:1000]


    def __del__(self):
        self.q.join()
        #self.P.terminate()
        #while len(self.metrics) > 0:
            #self.log_batch()

    @staticmethod
    def log_metric(key, value, step):
        metric = Metric(key=key, value=value, timestamp=int(time.time() * 1000), step=step)
        instance = Logger.getInstance()
        instance.q.put(metric)
        #instance.metrics.append(metric)

        #if len(instance.metrics) >= 1000:
            #instance.q.put(instance.metrics[:1000])
            #del instance.metrics[:1000]
        #mlflow.log_metric(key=f"func_{metric.key}", value=metric.value, step=metric.step)

        #inst = Logger.getInstance()
        #inst.metrics.append(metric)
        #if len(inst.metrics) > 1000:
            #inst.log_batch()
    

#import multiprocessing
#multiprocessing.Process(target=worker).start()