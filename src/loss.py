import jittor as jt

class EntropyLoss:
    def __init__(self, args):
        super(EntropyLoss, self).__init__()
        self.N_samples = args.N_rand
        self.type_ = 'log2'
        self.threshold = 0.1
        self.computing_entropy_all = False
        self.smoothing = False
        self.computing_ignore_smoothing = False
        self.entropy_log_scaling = False
        self.N_entropy = args.N_entropy 
        self.eps = 1e-10
    
    def _calc_entropy_loss(self, sigma, acc):
        ray_prob = sigma / (jt.sum(sigma,-1).unsqueeze(-1) + self.eps)
        entropy_ray = jt.sum(-ray_prob * jt.log2(ray_prob + self.eps), -1)
        
        entropy_ray *= (acc > self.threshold).detach() # ignore no hit position ray
        entropy_ray_loss = jt.mean(entropy_ray, -1)
        return entropy_ray_loss
    
    def ray(self, density, acc):
        acc = acc[self.N_samples:]
        density = density[self.N_samples:]
        density = jt.nn.relu(density[...,-1])
        sigma = 1 - jt.exp(-density)

        entropy_ray_loss = self._calc_entropy_loss(sigma, acc)
        return entropy_ray_loss

    def ray_zvals(self, sigma, acc):
        acc = acc[self.N_samples:]
        sigma = sigma[self.N_samples:]

        entropy_ray_loss = self._calc_entropy_loss(sigma, acc)
        return entropy_ray_loss

