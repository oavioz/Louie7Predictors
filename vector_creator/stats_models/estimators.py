import scipy.stats as cpy
import numpy as np
from statsmodels.robust.scale import qn_scale, Huber
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.regression.linear_model import yule_walker

'''
def skew(input_vector):
    return sp.skew(input_vector, bias=False)


def kurtosis(input_vector):
    return sp.kurtosis(input_vector)

'''


def entropy(input_vector):
    acc = np.sum(input_vector)
    prob_vec = list(map(lambda x: float(x/acc), input_vector))
    return cpy.entropy(prob_vec, base=2, axis=0)


'''
    :param ivi_matrix: IVI input matrix by rows for sample time 0..n-1  time frame
    :return: Irregularity score
    #I_u = np.array([timeOffsetInterval(u, Ui, W) for Ui in Un])
'''
def ivi_irregularity(ivi_matrix):
    l_row, l_col = ivi_matrix.shape
    c_var = []
    for i in range(0, l_col - 1):
        # get vector by column
        i_u = ivi_matrix[:,i]
        nz_count = np.count_nonzero(i_u)
        if not nz_count:
            c_var.append(float())
        else:
            norm_mean = np.mean(i_u)/np.std(i_u) #  (1/l_col) * np.sum(i_u, axis=0) / np.sqrt(1/(l_col-1) * np.sum(np.square(i_u - mean), axis=0))
            c_var.append(norm_mean)
    #return sum of scores devide by number of samples in time feame
    return [(np.sum(np.array(c_var))/l_row)]



def hober_m(data_set):
    huber = Huber(maxiter=100)(data_set)
    return [huber[0].item(0), huber[1].item(0)]


def qn(dataSet):
    return [qn_scale(dataSet)]



def kpss_trend_test(data):
    # dfDataSet is a pandas data frame
    # the null hypothesis is stationary time series data
    res = kpss(data, regression='c')
    # (statistics, p_value, critical values)
    return res


def yule_walker_test(data, order, yw_method='mle'):
    rho, sigma = yule_walker(data, order, method = yw_method)
    return rho, sigma

def ar_model(train, test, lag):   # train , test : numpy.ndarray
    # p_value  dicky_fuller_test hypotesis
    model = AutoReg(train, lag, old_names=False)
    model_fit = model.fit()
    coef = model_fit.params
    history = train[len(train) - lag:].tolist()
    predictions = list()
    for t in range(len(test)):
        history.append(test[t])
        history.remove(history[0])
        y_hat = coef[0]
        for d in range(lag):
            y_hat += coef[d+1] * history[lag-d-1]
        predictions.append(y_hat)
    mse = mean_squared_error(test, predictions)
    return np.sqrt(mse)

def ar_model_2(train, test, lag, mse):
    start = len(train)
    end = start + len(test) - 1
    model_fit = AutoReg(train, lag, old_names=False).fit()
    pred = model_fit.predict(start, end, dynamic=False)
    me = mean_squared_error(test, pred) if mse else mean_absolute_error(test,pred)
    return np.sqrt(me)

def timeOffsetInterval(u, Ui, W):
    def timeFrameOffset(u, Uk, W1):
        if (u >= 0 and u < Uk[0]):
            return Uk[0]
        elif (u > Uk[-1] and u < W1):
            return (W1 - Uk[-1])
        else:
            minUpper = list(filter(lambda x: x >= u, Uk))
            maxLower = list(filter(lambda x: x < u, Uk))
            return min(minUpper) - max(maxLower)

    return list(map(lambda x: timeFrameOffset(x, Ui, W), u))