
import datetime
import os


def get_savedir(model, dataset):
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        "LOG_DIR", date, dataset,
        model + dt.strftime('_%H_%M_%S')
    )
    os.makedirs(save_dir)
    return save_dir


def avg_both(mrs, mrrs, hits):
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h}


def format_metrics(metrics, split):
    result = "\t {} MR: {:.2f} | ".format(split, metrics['MR'])
    result += "MRR: {:.3f} | ".format(metrics['MRR'])
    result += "H@1: {:.3f} | ".format(metrics['hits@[1,3,10]'][0])
    result += "H@3: {:.3f} | ".format(metrics['hits@[1,3,10]'][1])
    result += "H@10: {:.3f}".format(metrics['hits@[1,3,10]'][2])
    return result


def write_metrics(writer, step, metrics, split):
    writer.add_scalar('{}_MR'.format(split), metrics['MR'], global_step=step)
    writer.add_scalar('{}_MRR'.format(split), metrics['MRR'], global_step=step)
    writer.add_scalar('{}_H1'.format(split), metrics['hits@[1,3,10]'][0], global_step=step)
    writer.add_scalar('{}_H3'.format(split), metrics['hits@[1,3,10]'][1], global_step=step)
    writer.add_scalar('{}_H10'.format(split), metrics['hits@[1,3,10]'][2], global_step=step)


def count_params(model):
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total
