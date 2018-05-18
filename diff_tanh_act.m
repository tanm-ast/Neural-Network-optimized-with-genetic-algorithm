function y = diff_tanh_act(x,a)
    y = 1-((tanh(x/a)).^2);
end