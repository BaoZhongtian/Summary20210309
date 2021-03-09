import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    total_line = []
    for index in range(50):
        data = numpy.genfromtxt(fname=os.path.joins('Result', 'Basic', 'Loss-%04d.csv' % index), dtype=float,
                                delimiter=',')
        total_line.extend(data)
    plt.plot(total_line)
    plt.xlabel('Batch Number')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss Function')
    plt.show()
