function[newW, newH] = normalize_WH(W, H, type)

    switch type
            
        case 'type1'            
            
            % normalizes rows in R so that the product LR stays the same
            % handle_zeros option: leaves rows that sum up to 0 as they were

            coeff_h = sum(H, 2);

            % added by HK
            coeff_h = max(coeff_h, 1e-16);

            coeff_w = 1 ./ coeff_h;
            left = diag(coeff_h);
            right = diag(coeff_w);
            newW = W * left;
            newH = right * H; 


        case 'type2'    
            % ported from normalizeWH.m from https://gitlab.com/ngillis/nmfbook/-/tree/master
            %
            % H^Te <= e entries in cols of H sum to at most 1
            Hn = SimplexProj( H );
            if norm(Hn - H) > 1e-3*norm(Hn)
               H = Hn; 
               % reoptimize W, because this normalization is NOT w.l.o.g. 
               options.inneriter = 100; 
               options.H = W'; 
               %W = nnls_PFGM(X',H',options);  % HK
               W = nnls_fpgm(V',H',options);   % HK
               
               W = W'; 
            end
            H = Hn; 

            newW = W;
            newH = H;            


        case 'type3' 
            % ported from normalizeWH.m from https://gitlab.com/ngillis/nmfbook/-/tree/master
            %            
            % He = e, entries in rows of H sum to 1

            scalH = sum(H');
            H = diag( scalH.^(-1) )*H;
            W = W*diag( scalH );

            newW = W;
            newH = H;   

        case 'type4'            
            % ported from normalizeWH.m from https://gitlab.com/ngillis/nmfbook/-/tree/master
            %
            % W^T e = e, entries in cols of W sum to 1

            scalW = sum(W);
            H = diag( scalW )*H;
            W = W*diag( scalW.^(-1) );

            newW = W;
            newH = H;      
            
        otherwise 

    end

end
        