################################################################################################################################################
### Project  : Master Degree research paper
### Script   : Degree on GIT.R
### Contents : A Rotation-based Boosting Algorithm With Adaptive Loss Functions
################################################################################################################################################

################################################################################################################################################
### Setting up environment
################################################################################################################################################
# Remove previous history
rm(list = ls())

# Load library
pkgs <- c("rpart", "ada", "caTools", "ggplot2", "reshape", "dplyr", "PairedData", "car", 
          "Matrix", "stringr", "randomForest", "ECoL", "FSA", "gridExtra", "coin")
sapply(pkgs, require, character.only = T)

# Load Datasets
load("datasets.RData")

# Load Functions
load("functions.RData")

################################################################################################################################################
### Analysis
################################################################################################################################################
## Figure 2-3 (a). Scatterplot of the artificial data before PCA 
pc.bf.rot <- prcomp(syn.df.tmp[, 1:2])
plot(syn.df.tmp[, 1], syn.df.tmp[, 2], xlab = 'x1', ylab = 'x2', type = 'n')
points(syn.df.tmp[y == 1, 1], syn.df.tmp[y == 1, 2], pch = 4, cex = 1.3, col = "black") 
points(syn.df.tmp[y == -1, 1], syn.df.tmp[y == -1, 2], pch = 19, cex = 1.3, col = "black")

## Figure 2-3 (b). Scatterplot of the artificial data after PCA 
rolex.tmp     <- RolexBoost(data = syn.df.tmp, n_feature = 2, rot_iter = 1, flex_iter = 1, bootstrap_rate = 1)
syn.df        <- rolex.tmp$df
pc.af.rot     <- prcomp(syn.df[, 1:2])
plot(syn.df[, 1], syn.df[, 2], xlab = 'x1', ylab = 'x2', type = 'n')
points(syn.df[y == 1, 1], syn.df[y == 1, 2], pch = 4, cex = 1.3, col = "black") 
points(syn.df[y == -1, 1], syn.df[y == -1, 2], pch = 19, cex = 1.3, col = "black")

## Figure 2-5. Classification error in relation to the exponential loss function
x   <- seq(-1.5, 1.5, 0.05)
l   <- 2
y.1 <- exp(-x)
plot(x, y.1, type = "l", col = "blue", ylim = c(0, 5), xlab = "yf", ylab = "loss", lwd = l, lty = 1, cex.lab = 1.5)
grid()
segments(-1.7, 1, 0, 1, lty = 3, col = "black", lwd = l)
segments( 0, 1, 0, 0, lty = 3, col = "black", lwd = l)
segments(-0, 0, 1.7, 0, lty = 3, col = "black", lwd = l)
legend(0.5, 5.0, legend = c("Exponential", "Classification(Zero-One)"),
       col = c("blue", "black"), lty = c(1, 3), cex = 1, lwd = l)

## Figure 2-7. Classification error in relation to the loss functions that are variated from the exponential loss
x   <- seq(-1.5, 1.5, 0.05)
s   <- 1.7
l   <- 2
y.1 <- exp(-x)
y.2 <- exp(-s*x)
y.3 <- exp(-(1/s)*x)
plot(x, y.1, type = "l", col = "blue", ylim = c(0, 5), xlab = "yf", ylab = "loss", lwd = l, lty = 1, cex.lab = 1.5)
grid()
par(new = TRUE)
plot(x, y.2, type = "l", col = "red", ylim = c(0, 5), xlab = "", ylab = "", lwd = l, lty = 4)
par(new = TRUE)
plot(x, y.3, type = "l", col = "green",  ylim = c(0, 5), xlab = "", ylab = "", lwd = l, lty = 5)
segments(-1.7, 1, 0, 1, lty = 3, col = "black", lwd = l)
segments( 0, 1, 0, 0, lty = 3, col = "black", lwd = l)
segments(-0, 0, 1.7, 0, lty = 3, col = "black", lwd = l)
legend(0.5, 5.0, legend = c("Exponential K > 1", "Exponential K = 1", "Exponential K < 1", "Classification (Zero-One)"),
       col = c("red", "blue", "green", "black"), lty = c(4, 1, 5, 3), cex = 1, lwd = l)

## Table 4-1. Description of the UCI datasets 
table.4.1.tmp <- matrix(NA, length(df.all), 2, dimnames = list(names(df.all), c("No.Instances", "No.Attributes")))
for (i in 1:nrow(table.4.1.tmp)){table.4.1.tmp[i,] <- dim(df.all[[i]]) - c(0, 1)}
table.4.1     <- table.4.1.tmp[-c(17:18, 20:21, 23:24, 26:27, 29:30, 32:34, 36:40),]
table.4.1     <- cbind(table.4.1, No.Classes = c(rep(2, 15), rep(3, 5)))

print(table.4.1)

## Table 4-2. Performance Benchmarks (average accuracy and rank) 
res.ranks <- as.matrix(res.acc.all[, c(3:9)])
for (i in 1:nrow(res.ranks)){res.ranks[i,] <- rank(-res.acc.all[i, c(3:9)], ties.method = "min")}
res.mrank           <- round(colMeans(unlist(res.ranks)), 2)
acc.mean            <- aggregate(x = res.acc.all[, c(3:9)], by = list(res.acc.all[, 2]), mean)
table4.2.tmp         <- rbind(acc.mean, lapply(acc.mean, mean), res.mrank[c(7, 1:6)])
table4.2.tmp[, 2:8]  <- round(table4.2.tmp[, 2:8], 4)
table4.2             <- table4.2.tmp[c(1, 5:7, 11:14, 18:20, 24:27, 2:4, 8:10, 15:17, 21:23, 28:32),]
table4.2[, 1]        <- c(na.omit(labels(df.all)), "Mean Accuracy", "Mean Rank")

print(table4.2)

## Table 4-3. Post-Hoc Test Results (p-value) 
res.fm.ph <- friedman.post.hoc(value ~ X2 | X1, data = melt(res.ranks))
res.dunn  <- dunnTest(melt(res.ranks)[, 3], melt(res.ranks)[, 2], method = 'bonferroni')$res
p.val.vec <- as.matrix(res.dunn[4])
res.p.val <- c(rep(NA, 7),                                              
               res.fm.ph$PostHoc.Test[2], rep(NA, 6),                    
               res.fm.ph$PostHoc.Test[c(6, 15)], rep(NA, 5),            
               res.fm.ph$PostHoc.Test[c(3, 12, 18)], rep(NA, 4),         
               res.fm.ph$PostHoc.Test[c(5, 14, 21, 17)], rep(NA, 3),     
               res.fm.ph$PostHoc.Test[c(1, 7, 11, 8, 10)], rep(NA, 2),  
               res.fm.ph$PostHoc.Test[c(4, 13, 20, 16, 19, 9)], NA,      # Nemenyi test
               p.val.vec[c(7, 9, 20, 10, 15, 8)], NA)                    # Bonferroni-Dunn test

table.4.3   <- as.data.frame(matrix(matrix(round(res.p.val, 3), 7, 8), nrow = 7, ncol = 8, 
                                    dimnames = list(c(names(res.mrank)), c(names(res.mrank), names(res.mrank)[7]))))
print(table.4.3)

## Figure 4-1. Ratio of each algorithm included in the Top-n rank on 30 UCI datasets
windowsFonts("Calibri" = windowsFont("Calibri"))
res.rank.ratio.tmp <- c()
for (i in 1:6){ res.rank.ratio.tmp <- cbind(res.rank.ratio.tmp, apply(res.ranks, 2, function(x){ length(which(x == i)) / length(x) })) }
res.rank.ratio.tmp <- matrix(res.rank.ratio.tmp, 42, 1)
for (i in 1:35){res.rank.ratio.tmp[i + 7,] <- res.rank.ratio.tmp[i + 7,] + res.rank.ratio.tmp[i,]}
res.rank.ratio     <- data.frame(Top_n     = c(rep("Top1", 7), rep("Top2", 7), rep("Top3", 7), rep("Top4", 7), rep("Top5", 7), rep("Top6", 7)),
                                 Algorithm = rep(c(1:7), 6),
                                 Ratio     = res.rank.ratio.tmp)
res.rank.ratio$adj.y            <- res.rank.ratio$Ratio
res.rank.ratio$adj.y[c(34, 40)] <- res.rank.ratio$adj.y[c(34, 40)] - 0.04

figure.4.1 <- ggplot(data = res.rank.ratio, aes(x = Top_n, y = Ratio, fill = factor(Algorithm))) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  scale_y_continuous(expand = c(0,0)) +
  coord_cartesian(ylim = c(0.0,1.05)) +
  theme_bw() +
  theme(legend.position = c(0.08, 0.85), legend.title = element_text(size = 10), 
        legend.text = element_text(size = 5)) +
  labs(x = "Top-n", y = "Ratio", fill = "Algorithm") +
  theme(axis.title.x = element_text(family = 'sans' , face = 2, color = 'black', size = 15)) +
  theme(axis.title.y = element_text(family = 'sans' , face = 2, color = 'black', size = 15)) +
  theme(axis.text.x  = element_text(family = 'sans' , face = 1, color = 'black', size = 13)) +
  theme(axis.text.y  = element_text(family = 'sans' , face = 1, color = 'black', size = 13)) +
  theme(plot.title   = element_text(family = 'sans' , face = 2, color = 'black', size = 13)) +
  theme(legend.text  = element_text(family = 'Calibri', size = 12)) +
  theme(legend.title = element_blank()) +
  geom_text(aes(y = adj.y, label = round(Ratio, 2)), color = "black", vjust = -0.5, position = position_dodge(0.69), family = 'Calibri', size = 4.7) +
  scale_fill_manual(values = c("lightgrey", "powderblue", "lightsteelblue", "khaki", "burlywood", "lightpink", "darkseagreen"),
                    labels = c("AdaBoost", "GentleBoost", "RotationForest", "RandomForest", "RotationBoost", "FlexBoost", "RolexBoost")) +
  guides(fill = guide_legend(keywidth = 0.2, keyheight = 0.2, default.unit = "inch"))

print(figure.4.1)

## Figure 5-2. Relationship between the degree of complexity change through rotation and the performance improvement from FlexBoost to RolexBoost observed in 30 UCI datasets. 
figure.5.2 <- ggplot(res.dis$uci, aes(x = res.dis$uci[ ,1], y = res.dis$uci[, 2], color = factor(res.dis$uci[, 3]))) + 
               geom_point() +
               labs(x = 'Degree of complexity change (in F3) through rotation', y = 'Performance improvement from FlexBoost to RolexBoost') +
               theme_bw() +
               theme(legend.text = element_text(size = 15), legend.title = element_blank(), 
               legend.position = c(0.12, 0.91), legend.background = element_rect(fill = "transparent", colour = "transparent")) +
               theme(axis.title.x = element_text(family = 'sans' , face = 2, color = 'black', size = 15)) +
               theme(axis.title.y = element_text(family = 'sans' , face = 2, color = 'black', size = 15)) +
               theme(axis.text.x  = element_text(family = 'sans' , face = 1, color = 'black', size = 13)) +
               theme(axis.text.y  = element_text(family = 'sans' , face = 1, color = 'black', size = 13)) +
               guides(colour = guide_legend(override.aes = list(size = 5), reverse = TRUE)) +
               geom_vline(xintercept = 0, color = 'grey')

print(figure.5.2)

## Figure 5-3. Relationship between the degree of complexity change through rotation and the performance improvement from FlexBoost to RolexBoost observed in 600 synthetic datasets. 
figure.5.3 <- ggplot(res.dis$art, aes(x = res.dis$art[ ,1], y = res.dis$art[, 2], color = factor(res.dis$art[, 3]))) + 
               geom_point() +
               labs(x = 'Degree of complexity change (in F3) through rotation', y = 'Performance improvement from FlexBoost to RolexBoost') +
               theme_bw() +
               theme(legend.text = element_text(size = 15), legend.title = element_blank(), 
               legend.position = c(0.12, 0.901), legend.background = element_rect(fill = "transparent", colour = "transparent")) +
               theme(axis.title.x = element_text(family = 'sans' , face = 2, color = 'black', size = 15)) +
               theme(axis.title.y = element_text(family = 'sans' , face = 2, color = 'black', size = 15)) +
               theme(axis.text.x  = element_text(family = 'sans' , face = 1, color = 'black', size = 13)) +
               theme(axis.text.y  = element_text(family = 'sans' , face = 1, color = 'black', size = 13)) +
               guides(colour = guide_legend(override.aes = list(size = 5), reverse = TRUE)) +
               geom_vline(xintercept = 0, color = 'grey') +
               scale_x_continuous(breaks = seq(-0.4, 0.4, 0.2)) +
               scale_y_continuous(breaks = seq(-0.05, 0.1, 0.05))

print(figure.5.3)

## Table 5-1. Performance Benchmarks (Optimal combination of L and T)
res.ranks <- as.matrix(res.acc.rolex[, c(3:11)])
for (i in 1:nrow(res.ranks)){res.ranks[i,] <- rank(-res.acc.rolex[i, c(3:11)], ties.method = "min")}
res.mrank             <- round(colMeans(unlist(res.ranks)), 2)
acc.mean              <- aggregate(x = res.acc.rolex[, c(3:11)], by = list(res.acc.rolex[, 2]), mean)
table.5.1.tmp         <- rbind(acc.mean, lapply(acc.mean, mean), res.mrank[c(9, 1:8)])
table.5.1.tmp[, 2:10] <- round(table.5.1.tmp[, 2:10], 4)
table.5.1             <- table.5.1.tmp[c(1, 5:7, 11:14, 18:20, 24:27, 2:4, 8:10, 15:17, 21:23, 28:32),]
table.5.1[, 1]        <- c(na.omit(labels(df.all)), "Mean Accuracy", "Mean Rank")

print(table.5.1)

