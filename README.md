[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6e56oMF4)

Task 4:

It makes sense that the alpha-beta algorithm is as successful and takes less time to compute than the minimax algorithm because it finds an equally optimal solution while looking at less states. Alpha stores the minimum score that the maximizing player is assured of and beta stores the maximum score that the minimizing player is assured of. If the maximizing player ever finds a value greater than the the max score that the minimizing player is assured of, they know the minimizer would never let this path occur, so it can be pruned, and the rest of this branch can be ignored. 

This strategy never prunes a branch that would be part of a solution, so it is just as optimal as minimax.This can be seen in my alpha_beta_performace graph, where both algorithms won and lost the same number of games as eachother. It looks at less states because some branches are pruned, so it will be more efficient and expand-less states which can be clearly seen in my states_expanded graph.

Side note: Somehow (and I genuinely don't know how this happened) I was working with the wrong stencil code when I build this (it was from 5 months ago). If anything seems weird, thats probably why. 