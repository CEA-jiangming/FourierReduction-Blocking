 function [matrix] = guessmatrix_test(diagonly, operator, rows, cols)
% Guesses the matrix corresponding to a given operator
% by operating on different delta vectors

maxnonzeros = rows*cols;
operdiag = zeros(rows,cols);

for i=1:rows
    for j=1:cols
        deltaim = zeros(rows,cols);
        deltaim(i,j) = 1;
        currcol = reshape(operator(deltaim),rows,cols);  % SLOW SLOW SLOW
        if diagonly
    %        if i > maxnonzeros
    %            break
    %        end
            operdiag(i,j) = currcol(i,j);
        else
%             matrix(:,i) = currcol;
            operdiag(i,j) = sum(currcol(:));
        end
        clear deltacol
    end
end
matrix = sparse(1:maxnonzeros, 1:maxnonzeros, operdiag(:), rows*cols, rows*cols, maxnonzeros);
end

