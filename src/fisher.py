import numpy as np
def euclid(x,y) -> float:
    return np.sum(np.square(x-y))

# X and Y and lists of points
def compute_fisher_two_classes(X,Y):
    mu_X = np.mean(X,axis=0)
    mu_Y = np.mean(Y,axis=0)
    mu = (mu_X + mu_Y) / 2.0
    Vb = euclid(mu_X,mu) + euclid(mu_Y,mu)
    Vw = 0
    for x in X:
        Vw += euclid(x,mu_X)
    for y in Y:
        Vw += euclid(y,mu_Y)
    return (Vb/Vw)

if __name__ == "__main__":
    # An example
    X = [[1,0],[2,3]]
    Y = [[-1,0],[-2,3]]
    print(compute_fisher_two_classes(X,Y))


