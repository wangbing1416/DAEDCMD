import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode


class ODEFunc(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.wt = nn.Linear(hidden_size, hidden_size)

    def forward(self, t, x):
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        x = self.wt(x)
        x = self.dropout_layer(x)
        x = F.relu(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=0.01, atol=0.001, method='dopri5', adjoint=False, terminal=False):
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out


class NeuralDynamics(nn.Module):
    def __init__(self, input_size, hidden_size, rtol=0.01, atol=0.001, method='dopri5'):
        super(NeuralDynamics, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True),
                                         nn.Tanh(),
                                         nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(ODEFunc(hidden_size),
                                             rtol=rtol, atol=atol, method=method)  # t is like  continuous depth
        self.output_layer = nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, vt, x):
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        x = self.input_layer(x)
        hvx = self.neural_dynamic_layer(vt, x)
        output = self.output_layer(hvx)
        return output.squeeze()
