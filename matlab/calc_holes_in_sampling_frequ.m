
for e=1:length(D.experiments) 
    if e == 4
        continue
    end
    for trials = 1:length(D.experiments{1,e}.scan) %{1,trials}
        %fprintf('%f\n', trials)
        for t = 1:length(D.experiments{1,e}.scan{1,trials}.scan_t)
            %fprintf('%f\n',D.experiments{1,e}.scan{1,trials}.scan_t(t))
            if D.experiments{1,e}.scan{1,trials}.scan_t(t) == 18703378
                fprintf('exp ')
                fprintf('%f\n', e)
                fprintf('trial ')
                fprintf('%f\n', trials)
                fprintf('t ')
                fprintf('%f\n', t)
                %fprintf('%f\n', '%f\n', '%f\n', e, trials, t);
                fprintf('Stop')
            end
        end
    end
end