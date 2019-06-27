import numpy as np

def estimate_affine(pts_s, pts_t):
    '''
    estimate affine martix，M * theta = b，calculate the least squares to get theta
    first 4 of theta resized(2x2)is A, t is the other 2 od theta
        | x1 y1 0  0  1 0 |       | a  |       | x1' |
        | 0  0  x1 y1 0 1 |       | b  |       | y1' |
        | x2 y2 0  0  1 0 |   *   | c  |   =   | x2' |
        | 0  0  x2 y2 0 1 |       | d  |       | y2' |
        | x3 y3 0  0  1 0 |       | tx |       | x3' |
        | 0  0  x3 y3 0 1 |       | ty |       | y3' |
        |------> M <------|   |-> theta <-|   |-> b <-|
    params:
        pts_s: coordinates of points in source image,(2,n)
        pts_t: coordinates of points in target image,(2,n)
    return:
        A: (2,2) of martix
        t: (2,) of martix '''
    pts_num = pts_s.shape[1]
    M = np.zeros((2 * pts_num, 6))
    for i in range(pts_num):
        temp = [[pts_s[0, i], pts_s[1, i], 0, 0, 1, 0],
                [0, 0, pts_s[0, i], pts_s[1, i], 0, 1]]
        M[2 * i: 2 * i + 2, :] = np.array(temp)
    b = pts_t.T.reshape((2 * pts_num, 1))
    # try:
    theta = np.linalg.lstsq(M, b)[0]
    A = theta[:4].reshape((2, 2))
    t = theta[4:]
    # except np.linalg.linalg.LinAlgError:
    #     A = None
    #     t = None
    return A, t

def estimate_transform(pts_s, pts_t):
    '''
    estimate rigid martix,a is cosx,b is sinx,tx,ty is translation
        | x1 y1  1 0 |       | a  |       | x1' |
        | y1 -x1 0 1 |       | b  |       | y1' |
        | x2 y2  1 0 |   *   | tx |   =  | x2' |
        | y2 -x2 0 1 |       | ty |      | y2' |
        |------> M <------|   |-> theta <-|   |-> b <-|
     params:
        pts_s: coordinates of points in source image,(2,n)
        pts_t: coordinates of points in target image,(2,n)
    return:
        A: (2,2) of martix
        t: (2,) of martix
    '''
    pts_num = pts_s.shape[1]
    M = np.zeros((2*pts_num, 4))
    for i in range(pts_num):
        temp = [[pts_s[0, i], pts_s[1, i], 1, 0],
                [pts_s[1, i],-pts_s[0, i], 0, 1]]
        M[2*i: 2*i+2, :] = np.array(temp)
    b = pts_t.T.reshape((2*pts_num, 1))
    # try:
    theta = np.linalg.lstsq(M, b)[0]
    A = np.array([[theta[0],theta[1]],
                  [-theta[1],theta[0]]]).reshape(2,2)
    t = theta[2:]
    return A, t
class Ransac():
    def __init__(self, K=3, threshold=1,n=0.9999):
        ''' K is point number，
        threshold is the judgment that whether thepoint is interiorpoint or not，
        ITER_NUM: the number of iterations'''
        self.K = K
        self.threshold = threshold
        self.n=n
        
    def residual_lengths(self, A, t, pts_s, pts_t):
        '''
        according to the A and t to get the new coordinates pts_e--pts_s 
        compute residual distance
        return:
            residual: according to pts_e ang pts_t compute residual distance
            residual2: according to pts_s ang pts_t compute residual distance
        '''
        if not(A is None) and not(t is None):
            pts_e = np.dot(A, pts_s) + t
            residual =np.linalg.norm(pts_e - pts_t,axis=0,ord=2)
            residual2 =np.linalg.norm(pts_e - pts_s,axis=0,ord=2)
            # diff_square = np.power(pts_e - pts_t, 2)
            # residual = np.sqrt(np.sum(diff_square, axis=0))
        else:
            residual = None
        return residual,residual2

    def ransac_fit(self, pts_s, pts_t):
        '''
        get the transform martix and the index of interior points
        params:
            pts_s: coordinates of points in source image,(2,n)
            pts_t: coordinates of points in target image,(2,n)
        return:
           A: (2,2) of martix
           t: (2,) of martix 
           inliers: the index of interior points
           threshold: the judgment that whether thepoint is interiorpoint or not
        '''

        inliers_num =3
        A = None
        t = None
        inliers = []
        point_num = pts_s.shape[1]
#        ITER_NUM = 200*point_num
#        if point_num<150:
#            ITER_NUM=500*point_num
        ITER_NUM=int(((np.log10(1-self.n))/(np.log10(1-0.05*0.05*0.05))))
#        if point_num<150:
#            ITER_NUM=5*int(((np.log10(1-self.n))/(np.log10(1-0.1*0.1*0.1))))
        threshold=30
        a=100
        if point_num < 5:
            print(" matches are found - %d/%d" % (point_num, 5))
            print("A is None,t is None residual is None")
            return A, t, inliers,threshold
        for i in range(ITER_NUM):
            idx = np.random.randint(0, point_num, (self.K, 1))
            A_tmp, t_tmp = estimate_affine(pts_s[:, idx], pts_t[:, idx])
            residual,residual2 = self.residual_lengths(A_tmp, t_tmp, pts_s, pts_t)
            if not(residual is None):
                inliers_tmp = np.where(residual < threshold)
                inliers_num_tmp = len(inliers_tmp[0])
                if inliers_num_tmp ==inliers_num:
                    residual2=residual2[inliers_tmp[0]]
                    re_std=np.std(residual2)
#                    residual_sorted=sorted(residual)[:inliers_num_tmp]
#                    re_std =np.std(residual_sorted)
                    if re_std <a:
                         a=re_std
#                         print("best="+str(a)+"---"+str(inliers_num))
                         inliers_num=inliers_num_tmp
                         inliers=sorted(inliers_tmp)
                         A=A_tmp
                         t=t_tmp  
                if inliers_num_tmp >inliers_num:
                    residual2=residual2[inliers_tmp[0]]
                    re_std=np.std(residual2)
#                    residual_sorted=sorted(residual)[:inliers_num_tmp]
#                    re_std =np.std(residual_sorted)
                    a=re_std 
                    inliers_num=inliers_num_tmp
                    inliers=sorted(inliers_tmp)
                    A=A_tmp
                    t=t_tmp
        return A, t, inliers,threshold

def affine_matrix(kp_s, kp_t):#, fit_pos):
    '''
    using RANSAC to screen the alignment points and get the martix of transform
    params: 
        kp_s: the coordinates of alignment points that in the source image
        kp_t: the coordinates of alignment points that in the target image
    return:
        M: the transform martix 
        inliers: the index of interior points
        threshold: the judgment that whether thepoint is interiorpoint or not'''
#    kp_s = kp_s[:, fit_pos[:, 0]]
#    kp_t = kp_t[:, fit_pos[:, 1]]
    ransac =Ransac()
    A, t, inliers,threshold = ransac.ransac_fit(kp_s, kp_t)
    if not (A is None) and not (t is None):
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]
        A, t =estimate_affine(kp_s, kp_t)
#        A, t =estimate_transform(kp_s, kp_t)
        M = np.hstack((A, t))
    else:
        M =  None
    return M,inliers,threshold

