from utils import Utils, GPUTools
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep

class FitnessEvaluate(object):

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)
        self.log.info('Finish the generation of python files')

    def evaluate(self):
        """
        load fitness from cache file
        """
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _acc = _map[_key]
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f'%(indi.id, _key, float(_acc), indi.acc))
                indi.acc = float(_acc)
        self.log.info('Total hit %d individuals for fitness'%(_count))

        has_evaluated_offspring = False
        for indi in self.individuals:
            temp=1
            if indi.acc < 0:

                has_evaluated_offspring = True
                time.sleep(60)
                gpu_id = GPUTools.detect_availabel_gpu_id()
                while gpu_id is None:
                    time.sleep(300)
                    if temp==1:
                        self.log.info('Waiting for free GPU...')
                        temp=2
                    gpu_id = GPUTools.detect_availabel_gpu_id()
                if gpu_id is not None:
                    file_name = indi.id
                    self.log.info('Begin to train %s'%(file_name))
                    module_name = 'scripts.%s'%(file_name)
                    if module_name in sys.modules.keys():
                        self.log.info('Module:%s has been loaded, delete it'%(module_name))
                        del sys.modules[module_name]
                        _module = importlib.import_module('.', module_name)
                    else:
                        _module = importlib.import_module('.', module_name)
                    _class = getattr(_module, 'RunModel')
                    cls_obj = _class()
                    p=Process(target=cls_obj.do_work, args=('%d'%(gpu_id), file_name,))
                    p.start()
            else:
                file_name = indi.id
                self.log.info('%s has inherited the fitness as %.5f, no need to evaluate'%(file_name, indi.acc))
                f = open('./populations/after_%s.txt'%(file_name[4:6]), 'a+')
                f.write('%s=%.5f\n'%(file_name, indi.acc))
                f.flush()
                f.close()




        if has_evaluated_offspring:
            all_finished = False
            while all_finished is not True:
                time.sleep(300)
                all_finished = GPUTools.all_gpu_available()

        if has_evaluated_offspring:
            file_name = './populations/after_%s.txt'%(self.individuals[0].id[4:6])
            assert os.path.exists(file_name) == True
            f = open(file_name, 'r')
            fitness_map = {}
            for line in f:
                if len(line.strip()) > 0:
                    line = line.strip().split('=')
                    fitness_map[line[0]] = float(line[1])
            f.close()
            for indi in self.individuals:
                if indi.acc == -1:
                    if indi.id not in fitness_map:
                        self.log.warn('The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds'%(indi.id, file_name))
                        sleep(120)
                    indi.acc = fitness_map[indi.id]
        else:
            self.log.info('None offspring has been evaluated')

        Utils.save_fitness_to_cache(self.individuals)

