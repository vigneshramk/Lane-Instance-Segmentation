def save_checkpoint(state, is_best=False, filename='checkpoint.h5'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.h5')