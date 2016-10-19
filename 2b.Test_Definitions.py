# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:20:53 2016

@author: naus010
"""

# Test calc_mismatch
#
#xtrue = np.array([np.random.randint(0,10) for i in range(2*nx)])
#x0 = np.array([xtrue[i] +3+ 5* (2*np.random.random() - 1) for i in range(2*nx)])
#y = obs_oper(ForwardModel(xtrue[:nx],xtrue[nx:]),loc_obsStandard)
#mismatch = calc_mismatch(x0,y,loc_obsStandard)
#J = calc_J(mismatch)
#print J

# Officlal Adjoint tests

def grad_test(x0,pert = 10**(-5)):
    nx = len(x0)/2
    x0 = np.array(x0)
    x_priorA = x0
    deriv = calc_dJdx(x0)
    dE,dC = deriv[:nx], deriv[nx:]
    J_prior = calc_J(x0)
    
    values = []
    for i in range(nx):
        pert_array = np.zeros(2*nx)
        pert_array[i] = pert
        x0_pert = x0 + pert_array
        
        predict = pert*deriv[i]
        J_post = calc_J(x0_pert)
        reduct = (J_post - J_prior)
        if predict == reduct == 0:
            val = 0
        else:
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
#        print 'For grid cell',i,'......'
#        if val <= 0.01:
#            print 'Gradient test passed :)'
#        else:
#            print 'Gradient test failed :('
        values.append(val)
    return np.array( values )*100


def adj_test(nx,nt):
    x_test = np.array([5*(2*np.random.random()-1) for i in range(2*nx)])
    E,Cstart = x_test[:nx],x_test[nx:]
    Mx_test = obs_oper( ForwardModel( E, Cstart,nt = ntA ))
    
    nobs = len(loc_obsA)
    y_test = np.array([[5*(2*np.random.random()-1) for j in range(nobs)] for i in range(nt)])
    MTy_test = AdjointModel(y_test*Oerror**2)
    
    dot1 = np.dot(x_test, MTy_test)
    dot2 = np.dot(Mx_test.flatten(), y_test.flatten())
    
    return abs((dot1-dot2)/dot1)*100
    
    if abs(dot1 - dot2) < abs(.01*dot1):
        print 'Adjoint test passed'
    else:
        print 'Adjoint test failed'
        
# ------------------------------------------------------------------
# Running the tests
ntA=500

run_grad_test = True
run_adj_test = True
if run_grad_test:
    # Gradient test 
    print 'Running grad test...'  
    
    start = time.time()
    # Preparing some data
    nt = 500
    xtrue = np.array([np.random.randint(0,10) for i in range(2*nx)])
    x_priorA = xtrue
    #x0_test = np.array([np.random.randint(-10,10) for i in range(2*nx)])
    x0_test = np.array([xtrue[i]+(1e-1)*(2*np.random.random()-1) for i in range(2*nx)])
    loc_obsA = [5,50,75]
    dataA = obs_oper(ForwardModel( xtrue[:nx], xtrue[nx:], nt = nt ))
    
    # The test
    val = grad_test(x0_test,pert=1e-5)
    end = time.time()
    
    print 'time grad:',end - start
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)

    
if run_adj_test:
    # Adjoint test
    print 'Running adjoint test...'
    
    start = time.time()
    t1,t2,t3 = [],[],[]
    nt = 500
    for i in range(2):
        t1.append(adj_test(100,nt))
    t1 = np.array(t1)
    end = time.time()
    
    print 'time adjoint:',end - start
    print 'Adj test mean error (%):' , round(t1.mean(),3) , 'and the std of the error:' , round(t1.std(),3)












