import warnings
import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs, _flat_to_shape, _mixed_norm, _all_callback_names, _all_adjoint_callback_names
_shape_to_flat = lambda t, *x: torch.cat([adj_.flatten() if len(t) == 0 else adj_.flatten(1) for adj_ in x], 0 if len(t) == 0 else 1)
torch.set_default_dtype(torch.float64)


class OdeintAdjointMethod(torch.autograd.Function):
    
    @staticmethod
    
    def forward(ctx, shapes, func, y_aug0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):
        
        
        ctx.shapes = shapes
       
        
        
        def augmented_func(t, y_aug):
          
            y_flat = func(t, _shape_to_flat((), *y_aug))
            return _flat_to_shape(y_flat, (), shapes)
      
        ctx.func = func
       
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None
        
        y0, w0, a = _flat_to_shape(y_aug0, (), shapes)
        
        init = (y0, w0,a)
        
            
        with torch.no_grad():
            
           
            ans = odeint( augmented_func, init, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)
           
            if event_fn is None:
                y = _shape_to_flat((len(t),), *ans)
                ctx.save_for_backward(t, y, *adjoint_params)
                ctx.shapes = shapes
                return y
            
            else:
                event_t, ans= ans
                y = _shape_to_flat((len(t),), *ans)
                ctx.save_for_backward(t, y, event_t, *adjoint_params)
                ctx.shapes = shapes
                
                return (event_t, y)

    

    @staticmethod
    def backward(ctx, *grad_y): 
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            event_mode = ctx.event_mode
            if event_mode:
                t, y, event_t,*adjoint_params = ctx.saved_tensors
                y, w, a = _flat_to_shape(y, (len(t),), ctx.shapes)
                _t = t
                t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                
                grad_y = grad_y[1]
                grad_y,grad_w,_ = _flat_to_shape(grad_y, (len(t),), ctx.shapes)
               
            else:
                t, y, *adjoint_params = ctx.saved_tensors 
               
                y, w, a = _flat_to_shape(y, (len(t),), ctx.shapes)
                
                
                grad_y = grad_y[0]
                
                grad_y,grad_w, _ = _flat_to_shape(grad_y, (len(t),), ctx.shapes)
                
                
                

            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################
            
            
            a_detach = a.detach()
            a_init = torch.zeros_like(a_detach[0])
            aug_state = [torch.zeros((), dtype=y.dtype, device=y.device),  y[-1], w[-1], a[-1], grad_y[-1], torch.zeros_like(grad_w[-1]), a_init] 
          
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params]) 
            
            ##################################
            #    Set up backward ODE func    #
            ##################################

            def augmented_dynamics(t, y_aug):
                
               
                y = y_aug[1]
                w = y_aug[2]
                a = y_aug[3]
                adj_y = y_aug[4]
                adj_w = y_aug[5]
                adj_a = y_aug[6]
              
               
                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)
                    w = w.detach().requires_grad_(True)
                   

                    a = a.detach().requires_grad_(True)
                   
                   
                    
                    y_aug_ = (y, w, a)
                    
                    dyaug_dt = func(t, _shape_to_flat((), *y_aug_))
                   
                    dy_dt, dw_dt, da_dt = _flat_to_shape(dyaug_dt, (), ctx.shapes)
                   
                    
                    vjp_t = None
                   
                    vjp_y, vjp_w, *vjp_params = torch.autograd.grad(
                        (dy_dt, dw_dt),
                        (y,w) + adjoint_params,
                        (-adj_y,-adj_w),
                        allow_unused = True,
                        retain_graph = True
                    )
                   
                    
                    n = w.shape[-1]
                    indices = torch.arange(0,n-1,2)

                   
                    adj_w_i = adj_w[...,indices]
                    adj_w_i1 = adj_w[...,indices+1]
                    w_i = w[...,indices]
                    w_i1 = w[...,indices+1]
                    
                    diagonal_sum = adj_w_i * w_i + adj_w_i1 * w_i1
                    sub_diagonal_diff = adj_w_i * w_i1 - adj_w_i1 * w_i
                   
                    vjp_a = -  torch.cat((diagonal_sum, sub_diagonal_diff),dim = -1).sum(0, keepdim=True)
                  

                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                vjp_w = torch.zeros_like(w) if vjp_w is None else vjp_w
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]
                vjp_a = torch.zeros_like(a) if vjp_a is None else vjp_a
               
                return (vjp_t, dy_dt, dw_dt, da_dt, vjp_y, vjp_w, vjp_a,*vjp_params)

            
            for callback_name, adjoint_callback_name in zip(_all_callback_names, _all_adjoint_callback_names):
                try:
                    callback = getattr(func, adjoint_callback_name)
                except AttributeError:
                    pass
                else:
                    setattr(augmented_dynamics, callback_name, callback)

            ##################################
            #       Solve sdjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                
                if t_requires_grad:
                    
                    y_aug_i = (y[i],w[i],a[i])
                    func_eval = func(t[i], y_aug_i)
                    
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t
               
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
               
                aug_state = [a[1] for a in aug_state]
                aug_state[1] = y[i - 1]
                aug_state[2] = w[i - 1]
                aug_state[3] = a[i - 1]
                aug_state[4] += grad_y[i - 1]
                aug_state[5] += grad_w[i - 1]
               

            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            if event_mode and t_requires_grad:
                time_vjps = torch.cat([time_vjps[0].reshape(-1), torch.zeros_like(_t[1:])])

            adj_y = aug_state[4]
            adj_w = aug_state[5]
            adj_a = aug_state[6]
            adj_params = aug_state[7:]
            adj_y_aug = _shape_to_flat((), adj_y, adj_w, adj_a)

        return (None, None,  adj_y_aug, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):

   
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')
   
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("if 'adjoint_method != method', then we can't infer 'options' from 'adjoint_options'。")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
       
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  

   
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
      
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

   
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)
    
  
    state_norm = options["norm"]
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if event_fn is None:
        y = ans
    else:
        event_t, y = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
      
        y = _flat_to_shape(y, (len(t),), shapes)

    if event_fn is None:
        return y
    else:
        if shapes is not None:
            return event_t, *y
        else:
            return event_t, y


def find_parameters(module):

    assert isinstance(module, nn.Module)


    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):

    
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        
        adjoint_options["norm"] = default_adjoint_norm
    else:
        
        try:
            adjoint_norm = adjoint_options['norm']
        except KeyError:
            
            adjoint_options['norm'] = default_adjoint_norm
        else:
            
            if adjoint_norm == 'seminorm':
                
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    
                    return max(t.abs(), state_norm(y), state_norm(adj_y))
                adjoint_options['norm'] = adjoint_seminorm
            else:
                
                if shapes is None:
                      
                    pass  
                else:
                  

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = _flat_to_shape(y, (), shapes)
                        adj_y = _flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))
                    adjoint_options['norm'] = _adjoint_norm
