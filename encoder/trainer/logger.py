import wandb


class Logger(object):
    def log_eval(self, eval_result, k, data_type):
        message = ''
        for metric in eval_result:
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
                wandb.log({"{}/{}@{}".format(data_type, metric, k[i]):eval_result[metric][i]})
            message += '] '

        print(message)