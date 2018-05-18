function y = diff_smoid(x,a)

    y = a*exp(-a*x) ./ ((1 + exp(-a*x)).^2);

end